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

//////// 1D

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

//////// 2D

namespace v0 {

template <typename T, typename FP, int YSEQ>
__forceinline__ __device__ void load_prequant_2d(
    T*       data,
    uint32_t dimx,
    uint32_t gix,
    uint32_t dimy,
    uint32_t giy_base,
    uint32_t stridey,
    FP       ebx2_r,
    T        center[YSEQ + 1]);

template <typename T, typename FP, int YSEQ>
__forceinline__ __device__ void predict_2d(T center[YSEQ + 1]);

template <typename T, typename EQ, int YSEQ>
__forceinline__ __device__ void quantize_write_2d(
    T        delta[YSEQ + 1],
    uint32_t dimx,
    uint32_t gix,
    uint32_t dimy,
    uint32_t giy_base,
    uint32_t stridey,
    int      radius,
    EQ*      quant,
    T*       outlier);

// decompression load
template <typename T, typename EQ, int YSEQ>
__forceinline__ __device__ void load_fuse_2d(
    EQ*      quant,
    T*       outlier,
    uint32_t dimx,
    uint32_t gix,
    uint32_t dimy,
    uint32_t giy_base,
    uint32_t stridey,
    int      radius,
    T        private_buffer[YSEQ]);

template <typename T, typename EQ, typename FP, int YSEQ>
__forceinline__ __device__ void block_scan_2d(  //
    T           thread_private[YSEQ],
    volatile T* intermediate,
    FP          ebx2);

template <typename T, int YSEQ>
__forceinline__ __device__ void decomp_write_2d(
    T        thread_private[YSEQ],
    uint32_t dimx,
    uint32_t gix,
    uint32_t dimy,
    uint32_t giy_base,
    uint32_t stridey,
    T*       xdata);

}  // namespace v0

//////// 3D

namespace v0 {

}

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

////////////////////////////////////////////////////////////////////////////////

//////// 1D

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
        T    delta       = cur - prev;
        bool quantizable = fabs(delta) < radius;
        T    candidate   = delta + radius;

        // otherwise, need to reset shared memory (to 0)
        shmem_quant[idx + threadIdx.x * SEQ]   = quantizable * static_cast<EQ>(candidate);
        shmem_outlier[idx + threadIdx.x * SEQ] = (not quantizable) * candidate;
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

        // otherwise, need to reset shared memory (to 0)
        shmem_quant[idx + threadIdx.x * SEQ]   = quantizable * UI_delta;
        shmem_outlier[idx + threadIdx.x * SEQ] = (not quantizable) * delta;
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

////////////////////////////////////////////////////////////////////////////////

//////// 2D

template <typename T, typename FP, int YSEQ>
__forceinline__ __device__ void parsz::cuda::__device::v0::load_prequant_2d(
    // clang-format off
    T*       data,
    uint32_t dimx, uint32_t gix,
    uint32_t dimy, uint32_t giy_base, uint32_t stridey,
    FP ebx2_r,
    T  center[YSEQ + 1]
    // clang-format on
)
{
    auto g_id = [&](auto iy) { return (giy_base + iy) * stridey + gix; };

    // use a warp as two half-warps
    // block_dim = (16, 2, 1) makes a full warp internally

#pragma unroll
    for (auto iy = 0; iy < YSEQ; iy++) {
        if (gix < dimx and giy_base + iy < dimy) center[iy + 1] = round(data[g_id(iy)] * ebx2_r);
    }
    auto tmp = __shfl_up_sync(0xffffffff, center[YSEQ], 16, 32);  // same-warp, next-16
    if (threadIdx.y == 1) center[0] = tmp;
}

template <typename T, typename FP, int YSEQ>
__forceinline__ __device__ void parsz::cuda::__device::v0::predict_2d(T center[YSEQ + 1])
{
    /*
       Lorenzo 2D (1-layer) illustration
                 NW N NE
       notation   W C E   "->" to predict
       --------  SW S SE

                normal data layout       |   considering register file
                col(k-1)    col(k)       |   thread(k-1)        thread(k)
                                         |
       r(i-1)  -west[i-1]  +center[i-1]  |  -center(k-1)[i-1]  +center(k)[i-1]
       r(i  )  +west[i]   ->center[i]    |  +center(k-1)[i]   ->center(k)[i]

       calculation
       -----------
       delta = center[i] - (center[i-1] + west[i] - west[i-1])
             = (center[i] - center[i-1]) - (west[i] - west[i-1])

       With center[i] -= center[i-1] and west[i] -= west[i-1],
       delta = center[i] - west[i]

       For thread(k),
       delta(k) = center(k)[i] - center(k-1)[i]
                = center(k)[i] - SHFL_UP(center(k)[i], 1, HALF_WARP)
     */

#pragma unroll
    for (auto i = YSEQ; i > 0; i--) {
        // with center[i-1] intact in this iteration
        center[i] -= center[i - 1];
        // within a halfwarp (32/2)
        auto west = __shfl_up_sync(0xffffffff, center[i], 1, 16);
        if (threadIdx.x > 0) center[i] -= west;  // delta
    }
    __syncthreads();
}

template <typename T, typename EQ, int YSEQ>
__forceinline__ __device__ void parsz::cuda::__device::v0::quantize_write_2d(
    // clang-format off
    T        delta[YSEQ + 1],
    uint32_t dimx,  uint32_t gix,
    uint32_t dimy,  uint32_t giy_base, uint32_t stridey,
    int      radius,
    EQ*      quant, 
    T*       outlier
    // clang-format on
)
{
    auto get_gid = [&](auto i) { return (giy_base + i) * stridey + gix; };

#pragma unroll
    for (auto i = 1; i < YSEQ + 1; i++) {
        auto gid = get_gid(i - 1);

        if (gix < dimx and giy_base + (i - 1) < dimy) {
            bool quantizable = fabs(delta[i]) < radius;
            T    candidate   = delta[i] + radius;

            // outlier array is not in sparse form in this version
            quant[gid]   = quantizable * static_cast<EQ>(candidate);
            outlier[gid] = (not quantizable) * candidate;
        }
    }
}

// load to thread-private array (fuse at the same time)
template <typename T, typename EQ, int YSEQ>
__forceinline__ __device__ void parsz::cuda::__device::v0::load_fuse_2d(
    // clang-format off
    EQ*      quant,
    T*       outlier,
    uint32_t dimx, uint32_t gix,
    uint32_t dimy, uint32_t giy_base, uint32_t stridey,
    int      radius,
    T        thread_private[YSEQ]
    // clang-format on
)
{
    auto get_gid = [&](auto iy) { return (giy_base + iy) * stridey + gix; };

#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
        auto gid = get_gid(i);
        // even if we hit the else branch, all threads in a warp hit the y-boundary simultaneously
        if (gix < dimx and (giy_base + i) < dimy)
            thread_private[i] = outlier[gid] + static_cast<T>(quant[gid]) - radius;  // fuse
        else
            thread_private[i] = 0;  // TODO set as init state?
    }
}

// partial-sum along y-axis, sequantially
// then, in-warp partial-sum along x-axis
template <typename T, typename EQ, typename FP, int YSEQ>
__forceinline__ __device__ void
parsz::cuda::__device::v0::block_scan_2d(T thread_private[YSEQ], volatile T* intermediate, FP ebx2)
{
    //       ------> gix (x)
    //
    //   |   t(0,0)       t(0,1)       t(0,2)       t(0,3)       ... t(0,f)
    //   |
    //   |   thp(0,0)[0]  thp(0,0)[0]  thp(0,0)[0]  thp(0,0)[0]
    //  giy  thp(0,0)[1]  thp(0,0)[1]  thp(0,0)[1]  thp(0,0)[1]
    //  (y)  |            |            |            |
    //       thp(0,0)[7]  thp(0,0)[7]  thp(0,0)[7]  thp(0,0)[7]
    //
    //   |   t(1,0)       t(1,1)       t(1,2)       t(1,3)       ... t(1,f)
    //   |
    //   |   thp(1,0)[0]  thp(1,0)[0]  thp(1,0)[0]  thp(1,0)[0]
    //  giy  thp(1,0)[1]  thp(1,0)[1]  thp(1,0)[1]  thp(1,0)[1]
    //  (y)  |            |            |            |
    //       thp(1,0)[7]  thp(1,0)[7]  thp(1,0)[7]  thp(1,0)[7]

    constexpr auto BLOCK = 16;

    for (auto i = 1; i < YSEQ; i++) thread_private[i] += thread_private[i - 1];
    // two-pass: store for cross-thread-private update
    // TODO shuffle up by 16 in the same warp
    if (threadIdx.y == 0) intermediate[threadIdx.x] = thread_private[YSEQ - 1];
    __syncthreads();
    // broadcast the partial-sum result from a previous segment
    if (threadIdx.y == 1) {
        auto tmp = intermediate[threadIdx.x];
#pragma unroll
        for (auto i = 0; i < YSEQ; i++) thread_private[i] += tmp;  // regression as pointer
    }
    // implicit sync as there is half-warp divergence

#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
        for (auto d = 1; d < BLOCK; d *= 2) {
            T n = __shfl_up_sync(0xffffffff, thread_private[i], d, 16);  // half-warp shuffle
            if (threadIdx.x >= d) thread_private[i] += n;
        }
        thread_private[i] *= ebx2;  // scale accordingly
    }
}

// write to DRAM
template <typename T, int YSEQ>
__forceinline__ __device__ void parsz::cuda::__device::v0::decomp_write_2d(
    // clang-format off
    T        thread_private[YSEQ],
    uint32_t dimx, uint32_t gix,
    uint32_t dimy, uint32_t giy_base, uint32_t stridey,
    T*       xdata
    // clang-format on
)
{
    auto get_gid = [&](auto iy) { return (giy_base + iy) * stridey + gix; };

#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
        auto gid = get_gid(i);
        if (gix < dimx and (giy_base + i) < dimy) xdata[gid] = thread_private[i];
    }
}

////////////////////////////////////////////////////////////////////////////////

//////// 3D
