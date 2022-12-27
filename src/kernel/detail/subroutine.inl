/**
 * @file subroutine.inl
 * @author Jiannan Tian
 * @brief subroutines of kernels
 * @version 0.4
 * @date 2022-12-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <stdint.h>
#include <type_traits>
#include "subsub.inl"

namespace parsz {
namespace cuda {
namespace __device {

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
__forceinline__ __device__ void predict_quantize__no_outlier_1d(  //
    T            private_buffer[SEQ],
    volatile EQ* shmem_quant,
    int          radius,
    T            prev = 0);

// compression pred-quant, method 2
template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void predict_quantize_1d(  //
    T            private_buffer[SEQ],
    volatile EQ* shmem_quant,
    volatile T*  shmem_outlier,
    int          radius,
    T            prev = 0);

// decompression pred-quant
template <typename T, int SEQ, int NTHREAD>
__forceinline__ __device__ void block_scan_1d(
    T           private_buffer[SEQ],
    T           ebx2,
    volatile T* exchange_in,
    volatile T* exchange_out,
    volatile T* shmem_buffer);

}  // namespace v0

namespace v1_pncodec {

template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void
predict_quantize__no_outlier_1d(T private_buffer[SEQ], volatile EQ* shmem_quant, int radius, T prev);

template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void
predict_quantize_1d(T private_buffer[SEQ], volatile EQ* shmem_quant, volatile T* shmem_outlier, int radius, T prev);

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
__forceinline__ __device__ void parsz::cuda::__device::v0::predict_quantize__no_outlier_1d(  //
    T            private_buffer[SEQ],
    volatile EQ* shmem_quant,
    int          radius,
    T            prev)
{
    auto quantize_1d = [&](T& cur, T& prev, uint32_t idx) {
        shmem_quant[idx + threadIdx.x * SEQ] = static_cast<EQ>(cur - prev);
    };

    if (FIRST_POINT) {  // i == 0
        quantize_1d(private_buffer[0], prev, 0);
    }
    else {
#pragma unroll
        for (auto i = 1; i < SEQ; i++) quantize_1d(private_buffer[i], private_buffer[i - 1], i);
        __syncthreads();
    }
}

template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void parsz::cuda::__device::v0::predict_quantize_1d(
    T            private_buffer[SEQ],
    volatile EQ* shmem_quant,
    volatile T*  shmem_outlier,
    int          radius,
    T            prev)
{
    auto quantize_1d = [&](T& cur, T& prev, uint32_t idx) {
        T    delta                             = cur - prev;
        bool quantizable                       = fabs(delta) < radius;
        T    candidate                         = delta + radius;
        shmem_outlier[idx + threadIdx.x * SEQ] = (1 - quantizable) * candidate;
        shmem_quant[idx + threadIdx.x * SEQ]   = quantizable * static_cast<EQ>(candidate);
    };

    if (FIRST_POINT) {  // i == 0
        quantize_1d(private_buffer[0], prev, 0);
    }
    else {
#pragma unroll
        for (auto i = 1; i < SEQ; i++) quantize_1d(private_buffer[i], private_buffer[i - 1], i);
        __syncthreads();
    }
}

// decompression pred-quant
template <typename T, int SEQ, int NTHREAD>
__forceinline__ __device__ void parsz::cuda::__device::v0::block_scan_1d(
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

// v1_pncodec: quantization code uses PN::encode
template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void parsz::cuda::__device::v1_pncodec::predict_quantize__no_outlier_1d(  //
    T            private_buffer[SEQ],
    volatile EQ* shmem_quant,
    int          radius,
    T            prev)
{
    constexpr auto BYTEWIDTH = sizeof(EQ);
    using UI                 = EQ;
    using I                  = typename parsz::typing::Int<BYTEWIDTH>::T;

    auto quantize_1d = [&](T& cur, T& prev, uint32_t idx) {
        UI UI_delta                          = PN<BYTEWIDTH>::encode(static_cast<I>(cur - prev));
        shmem_quant[idx + threadIdx.x * SEQ] = UI_delta;
    };

    if (FIRST_POINT) {  // i == 0
        quantize_1d(private_buffer[0], prev, 0);
    }
    else {
#pragma unroll
        for (auto i = 1; i < SEQ; i++) quantize_1d(private_buffer[i], private_buffer[i - 1], i);
        __syncthreads();
    }
}

template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void parsz::cuda::__device::v1_pncodec::predict_quantize_1d(
    T            private_buffer[SEQ],
    volatile EQ* shmem_quant,
    volatile T*  shmem_outlier,
    int          radius,
    T            prev)
{
    constexpr auto BYTEWIDTH = sizeof(EQ);
    using UI                 = EQ;
    using I                  = typename parsz::typing::Int<BYTEWIDTH>::T;

    auto quantize_1d = [&](T& cur, T& prev, uint32_t idx) {
        T    delta       = cur - prev;
        bool quantizable = fabs(delta) < radius;
        UI   UI_delta    = PN<BYTEWIDTH>::encode(static_cast<I>(delta));

        if (quantizable)
            shmem_quant[idx + threadIdx.x * SEQ] = UI_delta;
        else
            shmem_outlier[idx + threadIdx.x * SEQ] = delta;
    };

    if (FIRST_POINT) {  // i == 0
        quantize_1d(private_buffer[0], prev, 0);
    }
    else {
#pragma unroll
        for (auto i = 1; i < SEQ; i++) quantize_1d(private_buffer[i], private_buffer[i - 1], i);
        __syncthreads();
    }
}
