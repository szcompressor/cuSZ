/**
 * @file l21.cu_hip.inl
 * @author Jiannan Tian
 * @brief Dual-ErrCtrl Lorenzo method.
 * @version 0.2
 * @date 2021-01-16
 * (create) 2019-09-23; (release) 2020-09-20; (rev1) 2021-01-16; (rev2) 2021-02-20; (rev3) 2021-04-11
 * (rev4) 2021-04-30
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_KERNEL_LORENZO_CUH
#define CUSZ_KERNEL_LORENZO_CUH

#include <cstddef>
// #include "utils/err.hh"
// #include "utils/timer.hh"

#if __has_include(<cub/cub.cuh>)
// #pragma message __FILE__ ": (CUDA 11 onward), cub from system path"
#include <cub/cub.cuh>
#else
// #pragma message __FILE__ ": (CUDA 10 or earlier), cub from git submodule"
#include "../../third_party/cub/cub/cub.cuh"
#endif

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

#define TIX threadIdx.x
#define TIY threadIdx.y
#define TIZ threadIdx.z
#define BIX blockIdx.x
#define BIY blockIdx.y
#define BIZ blockIdx.z
#define BDX blockDim.x
#define BDY blockDim.y
#define BDZ blockDim.z

using DIM    = unsigned int;
using STRIDE = unsigned int;

namespace cusz {

/**
 * @brief compress-time 1D Lorenzo pred-quant kernel
 *
 * @tparam Data type of input data
 * @tparam ErrCtrl type of error-control code
 * @tparam FP type for internal floating-point processing
 * @tparam BLOCK block size
 * @tparam SEQ degree of sequentiality
 * @param data input
 * @param errctrl output 1
 * @param outlier output 2
 * @param len3 data length in 3D
 * @param stride3 data stride in 3D
 * @param radius quant-code radius
 * @param ebx2_r precalculated reciprocal of eb*2
 */
template <typename Data, typename ErrCtrl, typename FP = float, int BLOCK = 256, int SEQ = 4>
__global__ void
c_lorenzo_1d1l(Data* data, ErrCtrl* errctrl, Data* outlier, dim3 len3, dim3 stride3, int radius, FP ebx2_r);

/**
 * @brief compress-time 2D Lorenzo pred-quant kernel
 *
 * @tparam Data type of input data
 * @tparam ErrCtrl type of error-control code
 * @tparam FP type for internal floating-point processing
 * @tparam BLOCK block size
 * @tparam SEQ degree of sequentiality
 * @param data input
 * @param errctrl output 1
 * @param outlier output 2
 * @param len3 data length in 3D
 * @param stride3 data stride in 3D
 * @param radius quant-code radius
 * @param ebx2_r precalculated reciprocal of eb*2
 */
template <typename Data, typename ErrCtrl, typename FP = float>
__global__ void c_lorenzo_2d1l_16x16data_mapto16x2(
    Data*    data,
    ErrCtrl* errctrl,
    Data*    outlier,
    dim3     len3,
    dim3     stride3,
    int      radius,
    FP       ebx2_r);

/**
 * @brief compress-time 3D Lorenzo pred-quant kernel
 *
 * @tparam Data type of input data
 * @tparam ErrCtrl type of error-control code
 * @tparam FP type for internal floating-point processing
 * @tparam BLOCK block size
 * @tparam SEQ degree of sequentiality
 * @param data input
 * @param errctrl output 1
 * @param outlier output 2
 * @param len3 data length in 3D
 * @param stride3 data stride in 3D
 * @param radius quant-code radius
 * @param ebx2_r precalculated reciprocal of eb*2
 */
template <typename Data, typename ErrCtrl, typename FP = float>
__global__ void c_lorenzo_3d1l_32x8x8data_mapto32x1x8(
    Data*    data,
    ErrCtrl* errctrl,
    Data*    outlier,
    dim3     len3,
    dim3     stride3,
    int      radius,
    FP       ebx2_r);

/**
 * @brief decompress-time 1D Lorenzo pred-quant kernel
 *
 * @tparam Data type of input data
 * @tparam ErrCtrl type of error-control code
 * @tparam FP type for internal floating-point processing
 * @tparam BLOCK block size
 * @tparam SEQ degree of sequentiality
 * @param outlier input 1
 * @param quant input 2
 * @param xdata output
 * @param len3 data length in 3D
 * @param stride3 data stride in 3D
 * @param radius quant-code radius
 * @param ebx2 precalculated eb*2
 */
template <
    typename Data,
    typename ErrCtrl,
    typename FP = float,
    int BLOCK   = 256,
    int SEQ     = 8>
__global__ void x_lorenzo_1d1l(
    Data*    outlier,  //
    ErrCtrl* quant,
    Data*    xdata,
    dim3     len3,
    dim3     stride3,
    int      radius,
    FP       ebx2);

/**
 * @brief decompress-time 2D Lorenzo pred-quant kernel
 *
 * @tparam Data type of input data
 * @tparam ErrCtrl type of error-control code
 * @tparam FP type for internal floating-point processing
 * @tparam BLOCK block size
 * @tparam SEQ degree of sequentiality
 * @param outlier input 1
 * @param quant input 2
 * @param xdata output
 * @param len3 data length in 3D
 * @param stride3 data stride in 3D
 * @param radius quant-code radius
 * @param ebx2 precalculated eb*2
 */
template <typename Data, typename ErrCtrl, typename FP = float>
__global__ void x_lorenzo_2d1l_16x16data_mapto16x2(
    Data*    outlier,
    ErrCtrl* quant,
    Data*    xdata,
    dim3     len3,
    dim3     stride3,
    int      radius,
    FP       ebx2);

/**
 * @brief decompress-time 3D Lorenzo pred-quant kernel
 *
 * @tparam Data type of input data
 * @tparam ErrCtrl type of error-control code
 * @tparam FP type for internal floating-point processing
 * @tparam BLOCK block size
 * @tparam SEQ degree of sequentiality
 * @param outlier input 1
 * @param quant input 2
 * @param xdata output
 * @param len3 data length in 3D
 * @param stride3 data stride in 3D
 * @param radius quant-code radius
 * @param ebx2 precalculated eb*2
 */
template <typename Data, typename ErrCtrl, typename FP = float>
__global__ void x_lorenzo_3d1l_32x8x8data_mapto32x1x8(
    Data*    outlier,
    ErrCtrl* quant,
    Data*    xdata,
    dim3     len3,
    dim3     stride3,
    int      radius,
    FP       ebx2);

/**
 * @brief decompress-time 3D Lorenzo pred-quant kernel (variant)
 *
 * @tparam Data type of input data
 * @tparam ErrCtrl type of error-control code
 * @tparam FP type for internal floating-point processing
 * @tparam BLOCK block size
 * @tparam SEQ degree of sequentiality
 * @param outlier input 1
 * @param quant input 2
 * @param xdata output
 * @param len3 data length in 3D
 * @param stride3 data stride in 3D
 * @param radius quant-code radius
 * @param ebx2 precalculated eb*2
 */
template <typename Data, typename ErrCtrl, typename FP = float>
__global__ void x_lorenzo_3d1lvar_32x8x8data_mapto32x1x8(
    Data*    outlier,
    ErrCtrl* quant,
    Data*    xdata,
    dim3     len3,
    dim3     stride3,
    int      radius,
    FP       ebx2);

}  // namespace cusz

namespace {

/**
 * @brief (Original SZ/cuSZ design) 1D: separate delta by radius in to quant-code and outlier
 */
template <typename Data, typename ErrCtrl, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void pred1d_radius_separate(
    Data              thread_scope[SEQ],
    volatile Data*    shmem_data,
    volatile ErrCtrl* shmem_quant,
    int               radius,
    Data              from_last_stripe = 0)
{
    if CONSTEXPR (FIRST_POINT) {  // i == 0
        Data delta                 = thread_scope[0] - from_last_stripe;
        bool quantizable           = fabs(delta) < radius;
        Data candidate             = delta + radius;
        shmem_data[0 + TIX * SEQ]  = (1 - quantizable) * candidate;  // output; reuse data for outlier
        shmem_quant[0 + TIX * SEQ] = quantizable * static_cast<ErrCtrl>(candidate);
    }
    else {
#pragma unroll
        for (auto i = 1; i < SEQ; i++) {
            Data delta                 = thread_scope[i] - thread_scope[i - 1];
            bool quantizable           = fabs(delta) < radius;
            Data candidate             = delta + radius;
            shmem_data[i + TIX * SEQ]  = (1 - quantizable) * candidate;  // output; reuse data for outlier
            shmem_quant[i + TIX * SEQ] = quantizable * static_cast<ErrCtrl>(candidate);
        }
        __syncthreads();
    }
}

template <typename Data, typename FP, int NTHREAD, int SEQ>
__forceinline__ __device__ void load1d(
    Data*          data,
    unsigned int   dimx,
    unsigned int   id_base,
    volatile Data* shmem_data,
    Data           thread_scope[SEQ],
    Data&          from_last_stripe,
    FP             ebx2_r)
{
#pragma unroll
    for (auto i = 0; i < SEQ; i++) {
        auto id = id_base + TIX + i * NTHREAD;
        if (id < dimx) { shmem_data[TIX + i * NTHREAD] = round(data[id] * ebx2_r); }
    }
    __syncthreads();

    for (auto i = 0; i < SEQ; i++) thread_scope[i] = shmem_data[TIX * SEQ + i];

    if (TIX > 0) from_last_stripe = shmem_data[TIX * SEQ - 1];
    __syncthreads();
}

template <typename Data, typename ErrCtrl, int NTHREAD, int SEQ, bool NO_R_SEPARATE>
__forceinline__ __device__ void write1d(
    volatile Data*    shmem_data,
    Data*             data,
    unsigned int      dimx,
    unsigned int      id_base,
    volatile ErrCtrl* shmem_quant = nullptr,
    ErrCtrl*          quant       = nullptr)
{
#pragma unroll
    for (auto i = 0; i < SEQ; i++) {
        auto id = id_base + TIX + i * NTHREAD;
        if (id < dimx) {
            if CONSTEXPR (NO_R_SEPARATE) {  // TODO no-radius-separate uses shmem_data
                quant[id] = shmem_data[TIX + i * NTHREAD];
            }
            else {
                data[id]  = shmem_data[TIX + i * NTHREAD];
                quant[id] = shmem_quant[TIX + i * NTHREAD];
            }
        }
    }
}

template <typename Data, typename FP, int YSEQ>
__forceinline__ __device__ void load2d_prequant(
    Data*        data,
    Data         center[YSEQ + 1],
    unsigned int dimx,
    unsigned int dimy,
    unsigned int stridey,
    unsigned int gix,
    unsigned int giy_base,
    FP           ebx2_r)
{
    auto get_gid = [&](auto i) { return (giy_base + i) * stridey + gix; };

#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
        if (gix < dimx and giy_base + i < dimy) center[i + 1] = round(data[get_gid(i)] * ebx2_r);
    }
    auto tmp = __shfl_up_sync(0xffffffff, center[YSEQ], 16);  // same-warp, next-16
    if (TIY == 1) center[0] = tmp;
}

template <typename Data, typename FP, int YSEQ>
__forceinline__ __device__ void pred2d(Data center[YSEQ + 1])
{
    /* prediction
         original form:  Data delta = center[i] - center[i - 1] + west[i] - west[i - 1];
            short form:  Data delta = center[i] - west[i];
       */
#pragma unroll
    for (auto i = YSEQ; i > 0; i--) {
        center[i] -= center[i - 1];
        auto west = __shfl_up_sync(0xffffffff, center[i], 1, 16);
        if (TIX > 0) center[i] -= west;
    }
    __syncthreads();
}

template <typename Data, typename ErrCtrl, int YSEQ>
__forceinline__ __device__ void postquant_write2d(
    Data         center[YSEQ + 1],
    ErrCtrl*     quant,
    Data*        outlier,
    unsigned int dimx,
    unsigned int dimy,
    unsigned int stridey,
    int          radius,
    unsigned int gix,
    unsigned int giy_base)
{
    auto get_gid = [&](auto i) { return (giy_base + i) * stridey + gix; };

#pragma unroll
    for (auto i = 1; i < YSEQ + 1; i++) {
        auto gid = get_gid(i - 1);

        if (gix < dimx and giy_base + i - 1 < dimy) {
            bool quantizable = fabs(center[i]) < radius;
            Data candidate   = center[i] + radius;
            outlier[gid]     = (1 - quantizable) * candidate;  // output; reuse data for outlier
            quant[gid]       = quantizable * static_cast<ErrCtrl>(candidate);
        }
    }
}

}  // namespace

template <
    typename Data,
    typename ErrCtrl,
    typename FP,
    int BLOCK,
    int SEQ>
__global__ void cusz::c_lorenzo_1d1l(  //
    Data*    data,
    ErrCtrl* quant,
    Data*    outlier,
    dim3     len3,
    dim3     stride3,
    int      radius,
    FP       ebx2_r)
{
    constexpr auto NTHREAD = BLOCK / SEQ;

    __shared__ struct {
        union {
            uint8_t uninitialized[BLOCK * sizeof(Data) + BLOCK * sizeof(ErrCtrl)];
            Data    data[BLOCK];
        } space;
    } shmem;

    auto id_base = BIX * BLOCK;

    Data thread_scope[SEQ];
    Data from_last_stripe{0};

    /********************************************************************************
     * load from DRAM using striped layout, perform prequant
     ********************************************************************************/
    load1d<Data, FP, NTHREAD, SEQ>(data, len3.x, id_base, shmem.space.data, thread_scope, from_last_stripe, ebx2_r);

    // the original SZ/cuSZ design
    auto shmem_quant = reinterpret_cast<ErrCtrl*>(shmem.space.uninitialized + sizeof(Data) * BLOCK);
    pred1d_radius_separate<Data, ErrCtrl, SEQ, true>(
        thread_scope, shmem.space.data, shmem_quant, radius, from_last_stripe);
    pred1d_radius_separate<Data, ErrCtrl, SEQ, false>(thread_scope, shmem.space.data, shmem_quant, radius);
    write1d<Data, ErrCtrl, NTHREAD, SEQ, false>(shmem.space.data, outlier, len3.x, id_base, shmem_quant, quant);
}

template <typename Data, typename ErrCtrl, typename FP>
__global__ void cusz::c_lorenzo_2d1l_16x16data_mapto16x2(
    Data*    data,
    ErrCtrl* quant,
    Data*    outlier,
    dim3     len3,
    dim3     stride3,
    int      radius,
    FP       ebx2_r)
{
    constexpr auto BLOCK = 16;
    constexpr auto YSEQ  = 8;

    Data center[YSEQ + 1] = {0};  // nw  n
                                  //  w  center

    auto gix      = BIX * BDX + TIX;           // BDX == 16
    auto giy_base = BIY * BLOCK + TIY * YSEQ;  // BDY * YSEQ = BLOCK == 16

    load2d_prequant<Data, FP, YSEQ>(data, center, len3.x, len3.y, stride3.y, gix, giy_base, ebx2_r);
    pred2d<Data, FP, YSEQ>(center);
    postquant_write2d<Data, ErrCtrl, YSEQ>(center, quant, outlier, len3.x, len3.y, stride3.y, radius, gix, giy_base);
}

template <typename Data, typename ErrCtrl, typename FP>
__global__ void cusz::c_lorenzo_3d1l_32x8x8data_mapto32x1x8(
    Data*    data,
    ErrCtrl* quant,
    Data*    outlier,
    dim3     len3,
    dim3     stride3,
    int      radius,
    FP       ebx2_r)
{
    constexpr auto  BLOCK = 8;
    __shared__ Data shmem[8][8][32];

    auto z = TIZ;

    auto gix      = BIX * (BLOCK * 4) + TIX;
    auto giy_base = BIY * BLOCK;
    auto giz      = BIZ * BLOCK + z;
    auto base_id  = gix + giy_base * stride3.y + giz * stride3.z;

    /********************************************************************************
     * load from DRAM, perform prequant
     ********************************************************************************/
    if (gix < len3.x and giz < len3.z) {
        for (auto y = 0; y < BLOCK; y++) {
            if (giy_base + y < len3.y) {
                shmem[z][y][TIX] = round(data[base_id + y * stride3.y] * ebx2_r);  // prequant (fp presence)
            }
        }
    }
    __syncthreads();  // necessary to ensure correctness

    auto x = TIX % 8;

    for (auto y = 0; y < BLOCK; y++) {
        Data delta;

        /********************************************************************************
         * prediction
         ********************************************************************************/
        delta = shmem[z][y][TIX] - ((z > 0 and y > 0 and x > 0 ? shmem[z - 1][y - 1][TIX - 1] : 0)  // dist=3
                                    - (y > 0 and x > 0 ? shmem[z][y - 1][TIX - 1] : 0)              // dist=2
                                    - (z > 0 and x > 0 ? shmem[z - 1][y][TIX - 1] : 0)              //
                                    - (z > 0 and y > 0 ? shmem[z - 1][y - 1][TIX] : 0)              //
                                    + (x > 0 ? shmem[z][y][TIX - 1] : 0)                            // dist=1
                                    + (y > 0 ? shmem[z][y - 1][TIX] : 0)                            //
                                    + (z > 0 ? shmem[z - 1][y][TIX] : 0));                          //

        auto id = base_id + (y * stride3.y);

        bool quantizable = fabs(delta) < radius;
        Data candidate   = delta + radius;
        if (gix < len3.x and (giy_base + y) < len3.y and giz < len3.z) {
            outlier[id] = (1 - quantizable) * candidate;  // output; reuse data for outlier
            quant[id]   = quantizable * static_cast<ErrCtrl>(candidate);
        }
    }
    /* EOF */
}

template <typename Data, typename ErrCtrl, typename FP, int BLOCK, int SEQ>
__global__ void cusz::x_lorenzo_1d1l(  //
    Data*    outlier,
    ErrCtrl* quant,
    Data*    xdata,
    dim3     len3,
    dim3     stride3,
    int      radius,
    FP       ebx2)
{
    constexpr auto block_dim = BLOCK / SEQ;  // dividable

    // coalesce-load (warp-striped) and transpose in shmem (similar for store)
    typedef cub::BlockLoad<Data, block_dim, SEQ, cub::BLOCK_LOAD_WARP_TRANSPOSE>    BlockLoadT_outlier;
    typedef cub::BlockLoad<ErrCtrl, block_dim, SEQ, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT_quant;
    typedef cub::BlockStore<Data, block_dim, SEQ, cub::BLOCK_STORE_WARP_TRANSPOSE>  BlockStoreT_xdata;
    typedef cub::BlockScan<Data, block_dim, cub::BLOCK_SCAN_RAKING_MEMOIZE>
        BlockScanT_xdata;  // TODO autoselect algorithm

    __shared__ union TempStorage {  // overlap shared memory space
        typename BlockLoadT_outlier::TempStorage load_outlier;
        typename BlockLoadT_quant::TempStorage   load_quant;
        typename BlockStoreT_xdata::TempStorage  store_xdata;
        typename BlockScanT_xdata::TempStorage   scan_xdata;
    } temp_storage;

    // thread-scope tiled data
    union ThreadData {
        Data xdata[SEQ];
        Data outlier[SEQ];
    } thread_scope;
    ErrCtrl thread_scope_quant[SEQ];

    /********************************************************************************
     * load to thread-private array (fuse at the same time)
     * (BIX * BDX * SEQ) denotes the start of the data chunk that belongs to this thread block
     ********************************************************************************/
    BlockLoadT_quant(temp_storage.load_quant).Load(quant + (BIX * BDX) * SEQ, thread_scope_quant);
    __syncthreads();  // barrier for shmem reuse
    BlockLoadT_outlier(temp_storage.load_outlier).Load(outlier + (BIX * BDX) * SEQ, thread_scope.outlier);
    __syncthreads();  // barrier for shmem reuse

#pragma unroll
    for (auto i = 0; i < SEQ; i++) {
        auto id = (BIX * BDX + TIX) * SEQ + i;
        thread_scope.xdata[i] =
            id < len3.x ? thread_scope.outlier[i] + static_cast<Data>(thread_scope_quant[i]) - radius : 0;
    }
    __syncthreads();

    /********************************************************************************
     * perform partial-sum using cub::InclusiveSum
     ********************************************************************************/
    BlockScanT_xdata(temp_storage.scan_xdata).InclusiveSum(thread_scope.xdata, thread_scope.xdata);
    __syncthreads();  // barrier for shmem reuse

    /********************************************************************************
     * scale by ebx2 and write to DRAM
     ********************************************************************************/
#pragma unroll
    for (auto i = 0; i < SEQ; i++) thread_scope.xdata[i] *= ebx2;
    __syncthreads();  // barrier for shmem reuse

    BlockStoreT_xdata(temp_storage.store_xdata).Store(xdata + (BIX * BDX) * SEQ, thread_scope.xdata);
}

template <typename Data, typename ErrCtrl, typename FP>
__global__ void cusz::x_lorenzo_2d1l_16x16data_mapto16x2(
    Data*    outlier,
    ErrCtrl* quant,
    Data*    xdata,
    dim3     len3,
    dim3     stride3,
    int      radius,
    FP       ebx2)
{
    constexpr auto BLOCK = 16;
    constexpr auto YSEQ  = BLOCK / 2;  // sequentiality in y direction
    static_assert(BLOCK == 16, "In one case, we need BLOCK for 2D == 16");

    __shared__ Data intermediate[BLOCK];  // TODO use warp shuffle to eliminate this
    Data            thread_scope[YSEQ];
    /*
      .  ------> gix (x)
      |  t00    t01    t02    t03    ... t0f
      |  ts00_0 ts00_0 ts00_0 ts00_0
     giy ts00_1 ts00_1 ts00_1 ts00_1
     (y)  |      |      |      |
         ts00_7 ts00_7 ts00_7 ts00_7

      |  t10    t11    t12    t13    ... t1f
      |  ts00_0 ts00_0 ts00_0 ts00_0
     giy ts00_1 ts00_1 ts00_1 ts00_1
     (y)  |      |      |      |
         ts00_7 ts00_7 ts00_7 ts00_7
     */

    auto gix      = BIX * BLOCK + TIX;
    auto giy_base = BIY * BLOCK + TIY * YSEQ;  // BDY * YSEQ = BLOCK == 16
    auto get_gid  = [&](auto i) { return (giy_base + i) * stride3.y + gix; };

    /********************************************************************************
     * load to thread-private array (fuse at the same time)
     ********************************************************************************/
#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
        auto gid = get_gid(i);
        // even if we hit the else branch, all threads in a warp hit the y-boundary simultaneously
        if (gix < len3.x and giy_base + i < len3.y)
            thread_scope[i] = outlier[gid] + static_cast<Data>(quant[gid]) - radius;  // fuse
        else
            thread_scope[i] = 0;  // TODO set as init state?
    }

    /********************************************************************************
     * partial-sum along y-axis, sequantially
     ********************************************************************************/
    for (auto i = 1; i < YSEQ; i++) thread_scope[i] += thread_scope[i - 1];
    // two-pass: store for cross-threadscope update
    if (TIY == 0) intermediate[TIX] = thread_scope[YSEQ - 1];
    __syncthreads();
    // two-pass: load and update
    if (TIY == 1) {
        auto tmp = intermediate[TIX];
#pragma unroll
        for (auto& i : thread_scope) i += tmp;
    }

    /********************************************************************************
     * in-warp partial-sum along x-axis
     ********************************************************************************/
#pragma unroll
    for (auto& i : thread_scope) {
        for (auto d = 1; d < BLOCK; d *= 2) {
            Data n = __shfl_up_sync(0xffffffff, i, d, 16);
            if (TIX >= d) i += n;
        }
        i *= ebx2;
    }

    /********************************************************************************
     * write to DRAM
     ********************************************************************************/
#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
        auto gid = get_gid(i);
        if (gix < len3.x and giy_base + i < len3.y) xdata[gid] = thread_scope[i];
    }
}

template <typename Data, typename ErrCtrl, typename FP>
__global__ void cusz::x_lorenzo_3d1l_32x8x8data_mapto32x1x8(
    Data*    outlier,
    ErrCtrl* quant,
    Data*    xdata,
    dim3     len3,
    dim3     stride3,
    int      radius,
    FP       ebx2)
{
    constexpr auto BLOCK = 8;
    constexpr auto YSEQ  = BLOCK;
    static_assert(BLOCK == 8, "In one case, we need BLOCK for 3D == 8");

    __shared__ Data intermediate[BLOCK][4][8];
    Data            thread_scope[YSEQ];

    auto seg_id  = TIX / 8;
    auto seg_tix = TIX % 8;

    auto gix = BIX * (4 * BLOCK) + TIX, giy_base = BIY * BLOCK, giz = BIZ * BLOCK + TIZ;
    auto get_gid = [&](auto y) { return giz * stride3.z + (giy_base + y) * stride3.y + gix; };

    /********************************************************************************
     * load to thread-private array (fuse at the same time)
     ********************************************************************************/
#pragma unroll
    for (auto y = 0; y < YSEQ; y++) {
        auto gid = get_gid(y);
        if (gix < len3.x and giy_base + y < len3.y and giz < len3.z)
            thread_scope[y] = outlier[gid] + static_cast<Data>(quant[gid]) - static_cast<Data>(radius);  // fuse
        else
            thread_scope[y] = 0;
    }

    /********************************************************************************
     * partial-sum along y-axis, sequantially
     ********************************************************************************/
    for (auto y = 1; y < YSEQ; y++) thread_scope[y] += thread_scope[y - 1];

    /********************************************************************************
     * ND partial-sums along x- and z-axis
     * in-warp shuffle used: in order to perform, it's transposed after X-partial sum
     ********************************************************************************/
    auto dist = 1;
    Data addend;

#pragma unroll
    for (auto i = 0; i < BLOCK; i++) {
        Data val = thread_scope[i];

        for (dist = 1; dist < BLOCK; dist *= 2) {
            addend = __shfl_up_sync(0xffffffff, val, dist, 8);
            if (seg_tix >= dist) val += addend;
        }

        // x-z transpose
        intermediate[TIZ][seg_id][seg_tix] = val;
        __syncthreads();
        val = intermediate[seg_tix][seg_id][TIZ];
        __syncthreads();

        for (dist = 1; dist < BLOCK; dist *= 2) {
            addend = __shfl_up_sync(0xffffffff, val, dist, 8);
            if (seg_tix >= dist) val += addend;
        }

        intermediate[TIZ][seg_id][seg_tix] = val;
        __syncthreads();
        val = intermediate[seg_tix][seg_id][TIZ];
        __syncthreads();

        thread_scope[i] = val;
    }

    /********************************************************************************
     * write to DRAM
     ********************************************************************************/
#pragma unroll
    for (auto y = 0; y < YSEQ; y++) {
        if (gix < len3.x and giy_base + y < len3.y and giz < len3.z) { xdata[get_gid(y)] = thread_scope[y] * ebx2; }
    }
    /* EOF */
}

/********************************************************************************
 * experimental prototype toward further optmization
 ********************************************************************************/
template <typename Data, typename ErrCtrl, typename FP>
__global__ void cusz::x_lorenzo_3d1lvar_32x8x8data_mapto32x1x8(
    Data*    outlier,
    ErrCtrl* quant,
    Data*    xdata,
    dim3     len3,
    dim3     stride3,
    int      radius,
    FP       ebx2)
{
    constexpr auto BLOCK = 8;
    constexpr auto YSEQ  = BLOCK;
    static_assert(BLOCK == 8, "In one case, we need BLOCK for 3D == 8");

    __shared__ Data intermediate[BLOCK][4][8];
    Data            thread_scope = 0;

    auto seg_id  = TIX / 8;
    auto seg_tix = TIX % 8;

    auto gix = BIX * (4 * BLOCK) + TIX, giy_base = BIY * BLOCK, giz = BIZ * BLOCK + TIZ;
    auto get_gid = [&](auto y) { return giz * stride3.z + (giy_base + y) * stride3.y + gix; };

    auto y = 0;

    // even if we hit the else branch, all threads in a warp hit the y-boundary simultaneously
#pragma unroll
    for (y = 0; y < YSEQ; y++) {
        auto gid = get_gid(y);
        if (gix < len3.x and giy_base + y < len3.y and giz < len3.z)
            thread_scope += outlier[gid] + static_cast<Data>(quant[gid]) - static_cast<Data>(radius);  // fuse

        Data val = thread_scope;

        // shuffle, ND partial-sums
        for (auto dist = 1; dist < BLOCK; dist *= 2) {
            Data addend = __shfl_up_sync(0xffffffff, val, dist, 8);
            if (seg_tix >= dist) val += addend;
        }

        // x-z transpose
        intermediate[TIZ][seg_id][seg_tix] = val;
        __syncthreads();
        val = intermediate[seg_tix][seg_id][TIZ];
        __syncthreads();

        for (auto dist = 1; dist < BLOCK; dist *= 2) {
            Data addend = __shfl_up_sync(0xffffffff, val, dist, 8);
            if (seg_tix >= dist) val += addend;
        }

        intermediate[TIZ][seg_id][seg_tix] = val;
        __syncthreads();
        val = intermediate[seg_tix][seg_id][TIZ];
        __syncthreads();

        // thread_scope += val;

        if (gix < len3.x and giy_base + y < len3.y and giz < len3.z) { xdata[get_gid(y)] = val * ebx2; }
    }
}

#undef TIX
#undef TIY
#undef TIZ
#undef BIX
#undef BIY
#undef BIZ
#undef BDX
#undef BDY
#undef BDZ

#endif
