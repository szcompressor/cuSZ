/**
 * @file memops.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-12-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <stdint.h>
#include <type_traits>

namespace parsz {
namespace cuda {
namespace __device {

namespace wave32 {
template <typename T, int SEQ>
__forceinline__ __device__ void intrawarp_inclusivescan_1d(  //
    T private_buffer[SEQ]);

template <typename T, int SEQ, int NTHREAD>
__forceinline__ __device__ void intrablock_exclusivescan_1d(  //
    T           private_buffer[SEQ],
    volatile T* exchange_in,
    volatile T* exchange_out);
}  // namespace wave32

namespace v0 {

// compression load
template <typename T, typename FP, int NTHREAD, int SEQ>
__forceinline__ __device__ void load_prequant_1d(
    T*          data,
    uint32_t    dimx,
    uint32_t    id_base,
    volatile T* shmem,
    T           private_buffer[SEQ],
    T&          prev,
    FP          ebx2_r);

// decompression load
template <typename T, typename EQ, int NTHREAD, int SEQ>
__forceinline__ __device__ void load_fuse_1d(
    EQ*         quant,
    T*          outlier,
    uint32_t    dimx,
    uint32_t    id_base,
    int         radius,
    volatile T* shmem,
    T           private_buffer[SEQ]);

// compression and decompression store
template <typename T1, typename T2, int NTHREAD, int SEQ, bool NO_OUTLIER>
__forceinline__ __device__ void write_1d(  //
    volatile T1* shmem_a1,
    volatile T2* shmem_a2,
    uint32_t     dimx,
    uint32_t     id_base,
    T1*          a1,
    T2*          a2);

// compression pred-quant, method 1
template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void pq_no_outlier_1d(  //
    T            private_buffer[SEQ],
    volatile EQ* shmem_quant,
    int          radius,
    T            prev = 0);

// compression pred-quant, method 2
template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void pq_with_outlier_1d(  //
    T            private_buffer[SEQ],
    volatile EQ* shmem_quant,
    volatile T*  shmem_outlier,
    int          radius,
    T            prev = 0);

// decompression pred-quant
template <typename T, int SEQ, int NTHREAD>
__forceinline__ __device__ void blockwide_inclusivescan_1d(
    T           private_buffer[SEQ],
    T           ebx2,
    volatile T* exchange_in,
    volatile T* exchange_out,
    volatile T* shmem_buffer);

}  // namespace v0

namespace v1_pncodec {

template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void pq_no_outlier_1d(T private_buffer[SEQ], volatile EQ* shmem_quant, int radius, T prev);

template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void
pq_with_outlier_1d(T private_buffer[SEQ], volatile EQ* shmem_quant, volatile T* shmem_outlier, int radius, T prev);

}  // namespace v1_pncodec

}  // namespace __device
}  // namespace cuda
}  // namespace parsz

namespace parsz {
namespace typing {

// clang-format off
template <int BYTEWIDTH> struct Int;
template <> struct Int<1> { typedef int8_t  T; }; 
template <> struct Int<2> { typedef int16_t T; }; 
template <> struct Int<4> { typedef int32_t T; }; 
template <> struct Int<8> { typedef int64_t T; };

template <int BYTEWIDTH> struct UInt;
template <> struct UInt<1> { typedef uint8_t  T; }; 
template <> struct UInt<2> { typedef uint16_t T; }; 
template <> struct UInt<4> { typedef uint32_t T; }; 
template <> struct UInt<8> { typedef uint64_t T; };
// clang-format on

}  // namespace typing
}  // namespace parsz

template <int BYTEWIDTH>
struct PN {
    using UI = typename parsz::typing::UInt<BYTEWIDTH>::T;
    using I  = typename parsz::typing::Int<BYTEWIDTH>::T;

    // reference: https://lemire.me/blog/2022/11/25/making-all-your-integers-positive-with-zigzag-encoding/

    UI encode(I& x) { return (2 * x) ^ (x >> (BYTEWIDTH * 8 - 1)); }
    I  decode(UI& x) { return (x >> 1) ^ (-(x & 1)); }
};

template <typename T, typename FP, int NTHREAD, int SEQ>
__forceinline__ __device__ void parsz::cuda::__device::v0::load_prequant_1d(
    T*          data,
    uint32_t    dimx,
    uint32_t    id_base,
    volatile T* shmem,
    T           private_buffer[SEQ],
    T&          prev,  // TODO use pointer?
    FP          ebx2_r)
{
#pragma unroll
    for (auto i = 0; i < SEQ; i++) {
        auto id = id_base + threadIdx.x + i * NTHREAD;
        if (id < dimx) shmem[threadIdx.x + i * NTHREAD] = round(data[id] * ebx2_r);
    }
    __syncthreads();

#pragma unroll
    for (auto i = 0; i < SEQ; i++) private_buffer[i] = shmem[threadIdx.x * SEQ + i];
    if (threadIdx.x > 0) prev = shmem[threadIdx.x * SEQ - 1];
    __syncthreads();
}
template <typename T, typename EQ, int NTHREAD, int SEQ>
__forceinline__ __device__ void parsz::cuda::__device::v0::load_fuse_1d(
    EQ*         quant,
    T*          outlier,
    uint32_t    dimx,
    uint32_t    id_base,
    int         radius,
    volatile T* shmem,
    T           private_buffer[SEQ])
{
#pragma unroll
    for (auto i = 0; i < SEQ; i++) {
        auto local_id = threadIdx.x + i * NTHREAD;
        auto id       = id_base + local_id;
        if (id < dimx) shmem[local_id] = outlier[id] + static_cast<T>(quant[id]) - radius;
    }
    __syncthreads();

#pragma unroll
    for (auto i = 0; i < SEQ; i++) private_buffer[i] = shmem[threadIdx.x * SEQ + i];
    __syncthreads();
}

template <typename T1, typename T2, int NTHREAD, int SEQ, bool NO_OUTLIER>  // TODO remove NO_OUTLIER, use nullable
__forceinline__ __device__ void parsz::cuda::__device::v0::write_1d(
    volatile T1* shmem_a1,
    volatile T2* shmem_a2,
    uint32_t     dimx,
    uint32_t     id_base,
    T1*          a1,
    T2*          a2)
{
#pragma unroll
    for (auto i = 0; i < SEQ; i++) {
        auto id = id_base + threadIdx.x + i * NTHREAD;
        if (id < dimx) {
            if (NO_OUTLIER) {  //
                a1[id] = shmem_a1[threadIdx.x + i * NTHREAD];
            }
            else {
                a1[id] = shmem_a1[threadIdx.x + i * NTHREAD];
                a2[id] = shmem_a2[threadIdx.x + i * NTHREAD];
            }
        }
    }
}

template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void parsz::cuda::__device::v0::pq_no_outlier_1d(  //
    T            private_buffer[SEQ],
    volatile EQ* shmem_quant,
    int          radius,
    T            prev)
{
    if (FIRST_POINT) {  // i == 0
        T delta                            = private_buffer[0] - prev;
        shmem_quant[0 + threadIdx.x * SEQ] = static_cast<EQ>(delta);
    }
    else {
#pragma unroll
        for (auto i = 1; i < SEQ; i++) {
            T delta                            = private_buffer[i] - private_buffer[i - 1];
            shmem_quant[i + threadIdx.x * SEQ] = static_cast<EQ>(delta);
        }
        __syncthreads();
    }
}

template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void parsz::cuda::__device::v0::pq_with_outlier_1d(
    T            private_buffer[SEQ],
    volatile EQ* shmem_quant,
    volatile T*  shmem_outlier,
    int          radius,
    T            prev)
{
    if (FIRST_POINT) {  // i == 0
        T    delta                           = private_buffer[0] - prev;
        bool quantizable                     = fabs(delta) < radius;
        T    candidate                       = delta + radius;
        shmem_outlier[0 + threadIdx.x * SEQ] = (1 - quantizable) * candidate;  // output; reuse data for outlier
        shmem_quant[0 + threadIdx.x * SEQ]   = quantizable * static_cast<EQ>(candidate);
    }
    else {
#pragma unroll
        for (auto i = 1; i < SEQ; i++) {
            T    delta                           = private_buffer[i] - private_buffer[i - 1];
            bool quantizable                     = fabs(delta) < radius;
            T    candidate                       = delta + radius;
            shmem_outlier[i + threadIdx.x * SEQ] = (1 - quantizable) * candidate;  // output; reuse data for outlier
            shmem_quant[i + threadIdx.x * SEQ]   = quantizable * static_cast<EQ>(candidate);
        }
        __syncthreads();
    }
}

// decompression pred-quant
template <typename T, int SEQ, int NTHREAD>
__forceinline__ __device__ void parsz::cuda::__device::v0::blockwide_inclusivescan_1d(
    T           private_buffer[SEQ],
    T           ebx2,
    volatile T* exchange_in,
    volatile T* exchange_out,
    volatile T* shmem_buffer)
{
    namespace wave32 = parsz::cuda::__device::wave32;
    wave32::intrawarp_inclusivescan_1d<T, SEQ>(private_buffer);
    wave32::intrablock_exclusivescan_1d<T, SEQ, NTHREAD>(private_buffer, exchange_in, exchange_out);

    // put back to shmem
#pragma unroll
    for (auto i = 0; i < SEQ; i++) shmem_buffer[threadIdx.x * SEQ + i] = private_buffer[i] * ebx2;
    __syncthreads();
}

template <typename T, int SEQ>
__forceinline__ __device__ void parsz::cuda::__device::wave32::intrawarp_inclusivescan_1d(T private_buffer[SEQ])
{
    for (auto i = 1; i < SEQ; i++) private_buffer[i] += private_buffer[i - 1];
    T addend = private_buffer[SEQ - 1];

    // in-warp shuffle
    for (auto d = 1; d < 32; d *= 2) {
        T n = __shfl_up_sync(0xffffffff, addend, d, 32);
        if (threadIdx.x % 32 >= d) addend += n;
    }
    // exclusive scan
    T prev_addend = __shfl_up_sync(0xffffffff, addend, 1, 32);

    // propagate
    if (threadIdx.x % 32 > 0)
        for (auto i = 0; i < SEQ; i++) private_buffer[i] += prev_addend;
}

template <typename T, int SEQ, int NTHREAD>
__forceinline__ __device__ void parsz::cuda::__device::wave32::intrablock_exclusivescan_1d(
    T           private_buffer[SEQ],
    volatile T* exchange_in,
    volatile T* exchange_out)
{
    constexpr auto NWARP = NTHREAD / 32;
    static_assert(NWARP <= 32, "too big");

    auto warp_id = threadIdx.x / 32;
    auto lane_id = threadIdx.x % 32;

    if (lane_id == 31) exchange_in[warp_id] = private_buffer[SEQ - 1];
    __syncthreads();

    if (NWARP <= 8) {
        if (threadIdx.x == 0) {
            exchange_out[0] = 0;
            for (auto i = 1; i < NWARP; i++) exchange_out[i] = exchange_out[i - 1] + exchange_in[i - 1];
        }
    }
    else if (NWARP <= 32) {
        if (threadIdx.x <= 32) {
            auto addend = exchange_in[threadIdx.x];

            for (auto d = 1; d < 32; d *= 2) {
                T n = __shfl_up_sync(0xffffffff, addend, d, 32);
                if (threadIdx.x >= d) addend += n;
            }
            addend -= __shfl_sync(0xffffffff, addend, 0);
            exchange_out[warp_id] = addend;
        }
    }
    // else-case handled by static_assert
    __syncthreads();

    // propagate
    auto addend = exchange_out[warp_id];
    for (auto i = 0; i < SEQ; i++) private_buffer[i] += addend;
    __syncthreads();
};

// v1_pncodec: quantization code uses PN::encode
template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void parsz::cuda::__device::v1_pncodec::pq_no_outlier_1d(  //
    T            private_buffer[SEQ],
    volatile EQ* shmem_quant,
    int          radius,
    T            prev)
{
    constexpr auto BYTEWIDTH = sizeof(EQ);
    using UI                 = EQ;
    using I                  = typename parsz::typing::Int<BYTEWIDTH>::T;

    if (FIRST_POINT) {  // i == 0
        T  delta                           = private_buffer[0] - prev;
        UI UI_delta                        = PN<BYTEWIDTH>::encode(static_cast<I>(delta));
        shmem_quant[0 + threadIdx.x * SEQ] = UI_delta;
    }
    else {
#pragma unroll
        for (auto i = 1; i < SEQ; i++) {
            T  delta                           = private_buffer[i] - private_buffer[i - 1];
            UI UI_delta                        = PN<BYTEWIDTH>::encode(static_cast<I>(delta));
            shmem_quant[i + threadIdx.x * SEQ] = UI_delta;
        }
        __syncthreads();
    }
}

template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void parsz::cuda::__device::v1_pncodec::pq_with_outlier_1d(
    T            private_buffer[SEQ],
    volatile EQ* shmem_quant,
    volatile T*  shmem_outlier,
    int          radius,
    T            prev)
{
    constexpr auto BYTEWIDTH = sizeof(EQ);
    using UI                 = EQ;
    using I                  = typename parsz::typing::Int<BYTEWIDTH>::T;

    if (FIRST_POINT) {  // i == 0
        T    delta       = private_buffer[0] - prev;
        bool quantizable = fabs(delta) < radius;
        UI   UI_delta    = PN<BYTEWIDTH>::encode(static_cast<I>(delta));

        if (quantizable)
            shmem_quant[0 + threadIdx.x * SEQ] = UI_delta;
        else
            shmem_outlier[0 + threadIdx.x * SEQ] = delta;
    }
    else {
#pragma unroll
        for (auto i = 1; i < SEQ; i++) {
            T    delta       = private_buffer[i] - private_buffer[i - 1];
            bool quantizable = fabs(delta) < radius;
            UI   UI_delta    = PN<BYTEWIDTH>::encode(static_cast<I>(delta));

            if (quantizable)
                shmem_quant[i + threadIdx.x * SEQ] = UI_delta;
            else
                shmem_outlier[i + threadIdx.x * SEQ] = delta;
        }
        __syncthreads();
    }
}
