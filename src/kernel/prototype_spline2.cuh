/**
 * @file prototype_spline2.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-05-15
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_KERNEL_SPLINE2_CUH
#define CUSZ_KERNEL_SPLINE2_CUH

#include <stdint.h>
#include <stdio.h>
#include <type_traits>
#include "utils_spline.cuh"

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

#ifndef __CUDACC__
#define __global__
#define __device__
#define __host__
#define __shared__
#define __forceinline__
#endif

#define tix threadIdx.x
#define tiy threadIdx.y
#define tiz threadIdx.z
#define bix blockIdx.x
#define biy blockIdx.y
#define biz blockIdx.z
#define bdx blockDim.x
#define bdy blockDim.y
#define bdz blockDim.z

using DIM     = unsigned int;
using STRIDE  = unsigned int;
using DIM3    = dim3;
using STRIDE3 = dim3;

#define SHM_ERROR shm_errctrl

////////////////////////////////////////////////////////////////////////////////
// definition
////////////////////////////////////////////////////////////////////////////////

namespace kernel {

template <
    typename DataIter   = float*,
    typename QuantIter  = float*,
    typename FP         = float,
    int  NumThreads     = 256,
    bool DelayPostQuant = false,
    bool ProbePredError = false>
__global__ void c_spline3d_infprecis_32x8x8data(
    DataIter  data_first,
    DIM3      data_dim3,
    STRIDE3   data_stride3,
    QuantIter errctrl_first,
    DIM3      errctrl_dim3,
    STRIDE3   errctrl_stride3,
    FP        eb_r              = 1.0,
    FP        ebx2              = 2.0,
    int       radius            = 512,
    DataIter  pred_error        = nullptr,
    DataIter  compression_error = nullptr);

template <
    typename QuantIter  = float*,
    typename DataIter   = float*,
    typename FP         = float,
    int  NumThreads     = 256,
    bool DelayPostQuant = false>
__global__ void x_spline3d_infprecis_32x8x8data(
    QuantIter errctrl_first,    // input 1
    DIM3      errctrl_dim3,     //
    STRIDE3   errctrl_stride3,  //
    DataIter  anchor,           // input 2
    DIM3      anchor_dim3,      //
    STRIDE3   anchor_stride3,   //
    DataIter  data_first,       // output
    DIM3      data_dim3,        //
    STRIDE3   data_stride3,     //
    FP        eb_r   = 1.0,
    FP        ebx2   = 2.0,
    int       radius = 512);

}  // namespace kernel

#define Compress true
#define Decompress false

namespace kernel {
namespace internal {

template <typename T1, typename T2, typename FP, int NumThreads, bool Workflow = Compress, bool ProbPredError = false>
__device__ void spline3d_layout2_interpolate(
    volatile T1 shm_data[9][9][33],
    volatile T2 shm_errctrl[9][9][33],
    FP          eb_r   = 1.0,
    FP          ebx2   = 2.0,
    int         radius = 512);

}
}  // namespace kernel

template <bool Inclusive = true>
__forceinline__ __device__ bool xyz33x9x9_predicate(unsigned int x, unsigned int y, unsigned int z)
{
    if CONSTEXPR (Inclusive) {  //
        return x <= 32 and y <= 8 and z <= 8;
    }
    else {
        return x < 32 and y < 8 and z < 8;
    }
}

template <
    typename T1,
    typename T2,
    typename FP,
    typename LambdaX,
    typename LambdaY,
    typename LambdaZ,
    bool Blue,
    bool Yellow,
    bool Hollow,
    int  LinearBlockSize,
    int  BlockDimX,
    int  BlockDimY,
    bool Coarsen         = false,
    int  BlockDimZ       = 1,
    bool BorderInclusive = true,
    bool Workflow        = Compress>
__forceinline__ __device__ void interpolate_stage(
    volatile T1 shm_data[9][9][33],
    volatile T2 shm_errctrl[9][9][33],
    LambdaX     xmap,
    LambdaY     ymap,
    LambdaZ     zmap,
    int         unit,
    FP          eb_r,
    FP          ebx2,
    int         radius)
{
    static_assert(BlockDimX * BlockDimY * (Coarsen ? 1 : BlockDimZ) <= LinearBlockSize, "block oversized");
    static_assert((Blue or Yellow or Hollow) == true, "must be one hot");
    static_assert((Blue and Yellow) == false, "must be only one hot (1)");
    static_assert((Blue and Yellow) == false, "must be only one hot (2)");
    static_assert((Yellow and Hollow) == false, "must be only one hot (3)");

    auto run = [&](auto x, auto y, auto z) {
        if (xyz33x9x9_predicate<BorderInclusive>(x, y, z)) {
            T1 pred = 0;

            if CONSTEXPR (Blue) {  //
                pred = (shm_data[z - unit][y][x] + shm_data[z + unit][y][x]) / 2;
            }
            if CONSTEXPR (Yellow) {  //
                pred = (shm_data[z][y][x - unit] + shm_data[z][y][x + unit]) / 2;
            }
            if CONSTEXPR (Hollow) {  //
                pred = (shm_data[z][y - unit][x] + shm_data[z][y + unit][x]) / 2;
            }

            if CONSTEXPR (Workflow == Compress) {
                auto err = shm_data[z][y][x] - pred;
                // auto code                = kernel::internal::infprecis_quantcode(err, eb_r, radius);
                decltype(err) code;
                {
                    code = fabs(err) * eb_r + 1;
                    code = err < 0 ? -code : code;
                    code = int(code / 2) + radius;
                }
                shm_errctrl[z][y][x] = code;  // TODO double check if unsigned type works
                shm_data[z][y][x]    = pred + (code - radius) * ebx2;
            }
            else {
                auto code         = shm_errctrl[z][y][x];
                shm_data[z][y][x] = pred + (code - radius) * ebx2;
            }
        }
    };
    // -------------------------------------------------------------------------------- //

    if CONSTEXPR (Coarsen) {
        constexpr auto Total    = BlockDimX * BlockDimY * BlockDimZ;
        constexpr auto NumIters = Total / LinearBlockSize + 1;

        for (auto i = 0; i < NumIters; i++) {
            auto _tix = i * LinearBlockSize + tix;
            if (_tix < Total) {
                auto itix = (_tix % BlockDimX);
                auto itiy = (_tix / BlockDimX) % BlockDimY;
                auto itiz = (_tix / BlockDimX) / BlockDimY;
                auto x    = xmap(itix, unit);
                auto y    = ymap(itiy, unit);
                auto z    = zmap(itiz, unit);
                run(x, y, z);
            }
        }
    }
    else {
        auto itix = (tix % BlockDimX);
        auto itiy = (tix / BlockDimX) % BlockDimY;
        auto itiz = (tix / BlockDimX) / BlockDimY;
        auto x    = xmap(itix, unit);
        auto y    = ymap(itiy, unit);
        auto z    = zmap(itiz, unit);
        run(x, y, z);
    }
    __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// interpolation for layout2
////////////////////////////////////////////////////////////////////////////////

template <typename T1, typename T2, typename FP, int NumThreads, bool Workflow, bool ProbPredError>
__device__ void kernel::internal::spline3d_layout2_interpolate(
    volatile T1 shm_data[9][9][33],
    volatile T2 shm_errctrl[9][9][33],
    FP          eb_r,
    FP          ebx2,
    int         radius)
{
    auto xblue = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2); };
    auto yblue = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
    auto zblue = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2 + 1); };

    auto xyellow = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2 + 1); };
    auto yyellow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
    auto zyellow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };

    auto xhollow = [] __device__(int _tix, int unit) -> int { return unit * (_tix); };
    auto yhollow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2 + 1); };
    auto zhollow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };

    constexpr auto Coarsen         = true;
    constexpr auto NoCoarsen       = false;
    constexpr auto BorderInclusive = true;
    constexpr auto BorderExclusive = false;

    int unit = 4;

    // iteration 1
    interpolate_stage<
        T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
        true, false, false, NumThreads, 5, 2, NoCoarsen, 1, BorderInclusive, Workflow>(
        shm_data, shm_errctrl, xblue, yblue, zblue, unit, eb_r, ebx2, radius);
    interpolate_stage<
        T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, NumThreads, 4, 2, NoCoarsen, 3, BorderInclusive, Workflow>(
        shm_data, shm_errctrl, xyellow, yyellow, zyellow, unit, eb_r, ebx2, radius);
    interpolate_stage<
        T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
        false, false, true, NumThreads, 9, 1, NoCoarsen, 3, BorderInclusive, Workflow>(
        shm_data, shm_errctrl, xhollow, yhollow, zhollow, unit, eb_r, ebx2, radius);

    unit = 2;

    // iteration 2, TODO switch y-z order
    interpolate_stage<
        T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
        true, false, false, NumThreads, 9, 3, NoCoarsen, 2, BorderInclusive, Workflow>(
        shm_data, shm_errctrl, xblue, yblue, zblue, unit, eb_r, ebx2, radius);
    interpolate_stage<
        T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, NumThreads, 8, 3, NoCoarsen, 5, BorderInclusive, Workflow>(
        shm_data, shm_errctrl, xyellow, yyellow, zyellow, unit, eb_r, ebx2, radius);
    interpolate_stage<
        T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
        false, false, true, NumThreads, 17, 2, NoCoarsen, 5, BorderInclusive, Workflow>(
        shm_data, shm_errctrl, xhollow, yhollow, zhollow, unit, eb_r, ebx2, radius);

    unit = 1;

    // iteration 3
    interpolate_stage<
        T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
        true, false, false, NumThreads, 17, 5, Coarsen, 4, BorderInclusive, Workflow>(
        shm_data, shm_errctrl, xblue, yblue, zblue, unit, eb_r, ebx2, radius);
    interpolate_stage<
        T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, NumThreads, 16, 5, Coarsen, 9, BorderInclusive, Workflow>(
        shm_data, shm_errctrl, xyellow, yyellow, zyellow, unit, eb_r, ebx2, radius);
    /******************************************************************************
     test only: last step inclusive
     ******************************************************************************/
    // interpolate_stage<
    //     T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
    //     false, false, true, NumThreads, 33, 4, Coarsen, 9, BorderInclusive, Workflow>(
    //     shm_data, shm_errctrl, xhollow, yhollow, zhollow, unit, eb_r, ebx2, radius);
    /******************************************************************************
     production
     ******************************************************************************/
    interpolate_stage<
        T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
        false, false, true, NumThreads, 32, 4, Coarsen, 8, BorderExclusive, Workflow>(
        shm_data, shm_errctrl, xhollow, yhollow, zhollow, unit, eb_r, ebx2, radius);

    /******************************************************************************
     test only: print a block
     ******************************************************************************/
    // if (tix == 0 and bix == 0 and biy == 0 and biz == 0) { spline3d_print_block_from_GPU(shm_errctrl); }
    // if (tix == 0 and bix == 0 and biy == 0 and biz == 0) { spline3d_print_block_from_GPU(shm_data); }
}

////////////////////////////////////////////////////////////////////////////////
// API level kernel: inifite-precision spline3d, layout1
////////////////////////////////////////////////////////////////////////////////

template <
    typename DataIter,  //
    typename QuantIter,
    typename FP,
    int  NumThreads,
    bool DelayPostQuant,
    bool ProbePredError>
__global__ void kernel::c_spline3d_infprecis_32x8x8data(
    DataIter  data_first,
    DIM3      data_dim3,
    STRIDE3   data_stride3,
    QuantIter errctrl_first,
    DIM3      errctrl_dim3,
    STRIDE3   errctrl_stride3,
    FP        eb_r,
    FP        ebx2,
    int       radius,
    DataIter  pred_error,
    DataIter  compression_error)
{
    // compile time variables
    using Data  = typename std::remove_pointer<DataIter>::type;
    using Quant = typename std::remove_pointer<QuantIter>::type;

    if CONSTEXPR (ProbePredError) {
        //
    }
    else {
        __shared__ struct {
            Data  data[9][9][33];
            Quant errctrl[9][9][33];
        } shmem;

        kernel::internal::memops::c_spline3d_reset_scratch_33x9x9data<Data, Quant, NumThreads>(
            shmem.data, shmem.errctrl);
        kernel::internal::memops::spline3d_global2shmem_33x9x9data<Data, NumThreads>(
            data_first, shmem.data, data_dim3, data_stride3);
        kernel::internal::spline3d_layout2_interpolate<Data, Quant, FP, NumThreads, Compress, false>(
            shmem.data, shmem.errctrl, eb_r, ebx2, radius);
        kernel::internal::memops::spline3d_shmem2global_32x8x8data<Quant, NumThreads>(
            shmem.errctrl, errctrl_first, errctrl_dim3, errctrl_stride3);
    }
}

template <
    typename QuantIter,
    typename DataIter,
    typename FP,
    int  NumThreads,
    bool DelayPostQuant>
__global__ void kernel::x_spline3d_infprecis_32x8x8data(
    QuantIter errctrl_first,    // input 1
    DIM3      errctrl_dim3,     //
    STRIDE3   errctrl_stride3,  //
    DataIter  anchor,           // input 2
    DIM3      anchor_dim3,      //
    STRIDE3   anchor_stride3,   //
    DataIter  data_first,       // output
    DIM3      data_dim3,        //
    STRIDE3   data_stride3,     //
    FP        eb_r,
    FP        ebx2,
    int       radius)
{
    // compile time variables
    using Quant = typename std::remove_pointer<QuantIter>::type;
    using Data  = typename std::remove_pointer<DataIter>::type;

    __shared__ struct {
        Quant errctrl[9][9][33];
        Data  data[9][9][33];
    } shmem;

    kernel::internal::memops::x_spline3d_reset_scratch_33x9x9data<Data, Quant, NumThreads>(
        shmem.data, shmem.errctrl, anchor, anchor_dim3, anchor_stride3, radius);
    kernel::internal::memops::spline3d_global2shmem_33x9x9data<Quant, NumThreads>(
        errctrl_first, shmem.errctrl, errctrl_dim3, errctrl_stride3);
    kernel::internal::spline3d_layout2_interpolate<Data, Quant, FP, NumThreads, Decompress, false>(
        shmem.data, shmem.errctrl, eb_r, ebx2, radius);
    kernel::internal::memops::spline3d_shmem2global_32x8x8data<Data, NumThreads>(
        shmem.data, data_first, data_dim3, data_stride3);
}

#endif
