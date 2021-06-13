/**
 * @file spline.h
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-05-15
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_KERNEL_SPLINE_H
#define CUSZ_KERNEL_SPLINE_H

#include <stdint.h>
#include <stdio.h>
#include <type_traits>

#define COMPRESS true
#define DECOMPRESS false

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

#define TIX threadIdx.x
#define TIY threadIdx.y
#define TIZ threadIdx.z
#define BIX blockIdx.x
#define BIY blockIdx.y
#define BIZ blockIdx.z
#define BDX blockDim.x
#define BDY blockDim.y
#define BDZ blockDim.

using DIM     = unsigned int;
using STRIDE  = unsigned int;
using DIM3    = dim3;
using STRIDE3 = dim3;

constexpr int BLOCK8  = 8;
constexpr int BLOCK32 = 32;

#define SHM_ERROR shm_errctrl

////////////////////////////////////////////////////////////////////////////////
// definition
////////////////////////////////////////////////////////////////////////////////

template <
    typename DataIter      = float*,
    typename QuantIter     = float*,
    typename FP            = float,
    int  LINEAR_BLOCK_SIZE = 256,
    bool DELAY_POSTQUANT   = false,
    bool PROBE_PRED_ERROR  = false>
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
    typename QuantIter     = float*,
    typename DataIter      = float*,
    typename FP            = float,
    int  LINEAR_BLOCK_SIZE = 256,
    bool DELAY_POSTQUANT   = false>
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
    FP        eb_r,
    FP        ebx2,
    int       radius);

/********************************************************************************
 * device API
 ********************************************************************************/
template <
    typename T1,
    typename T2,
    typename FP,
    int  LINEAR_BLOCK_SIZE,
    bool WORKFLOW         = COMPRESS,
    bool PROBE_PRED_ERROR = false>
__device__ void spline3d_layout2_interpolate(
    volatile T1 shm_data[9][9][33],
    volatile T2 shm_errctrl[9][9][33],
    FP          eb_r,
    FP          ebx2,
    int         radius);

// template <
//     typename T1,
//     typename T2,
//     typename FP,
//     typename LambdaX,
//     typename LambdaY,
//     typename LambdaZ,
//     bool Blue,
//     bool Yellow,
//     bool Hollow,
//     int  LinearBlockSize,
//     int  BlockDimX,
//     int  BlockDimY,
//     bool Coarsen         = false,
//     int  BlockDimZ       = 1,
//     bool BorderInclusive = true,
//     bool WORKFLOW        = COMPRESS>
// __forceinline__ __device__ void interpolate_stage(
//     volatile T1 shm_data[9][9][33],
//     volatile T2 shm_errctrl[9][9][33],
//     LambdaX     xmap,
//     LambdaY     ymap,
//     LambdaZ     zmap,
//     int         unit,
//     FP          eb_r,
//     FP          ebx2,
//     int         radius);

////////////////////////////////////////////////////////////////////////////////
// interpolation for layout2
////////////////////////////////////////////////////////////////////////////////

/********************************************************************************
 * helper function
 ********************************************************************************/

namespace {

template <bool INCLUSIVE = true>
__forceinline__ __device__ bool xyz33x9x9_predicate(unsigned int x, unsigned int y, unsigned int z)
{
    if CONSTEXPR (INCLUSIVE) {  //
        return x <= 32 and y <= 8 and z <= 8;
    }
    else {
        return x < 32 and y < 8 and z < 8;
    }
}

// control block_id3 in function call
template <typename T, bool PRINT_FP = false, int XEND = 33, int YEND = 9, int ZEND = 9>
__device__ void spline3d_print_block_from_GPU(
    T volatile a[9][9][33],
    int  radius        = 512,
    bool compress      = true,
    bool print_errctrl = true)
{
    for (auto z = 0; z < ZEND; z++) {
        printf("\nprint from GPU, z=%d\n", z);
        printf("    ");
        for (auto i = 0; i < 33; i++) printf("%3d", i);
        printf("\n");

        for (auto y = 0; y < YEND; y++) {
            printf("y=%d ", y);
            for (auto x = 0; x < XEND; x++) {  //
                if CONSTEXPR (PRINT_FP) { printf("%.2e\t", (float)a[z][y][x]); }
                else {
                    T c = print_errctrl ? a[z][y][x] - radius : a[z][y][x];
                    if CONSTEXPR (compress) {
                        if (c == 0) { printf("%3c", '.'); }
                        else {
                            if (abs(c) >= 10) { printf("%3c", '*'); }
                            else {
                                if (print_errctrl) { printf("%3d", c); }
                                else {
                                    printf("%4.2f", c);
                                }
                            }
                        }
                    }
                    else {
                        if (print_errctrl) { printf("%3d", c); }
                        else {
                            printf("%4.2f", c);
                        }
                    }
                }
            }
            printf("\n");
        }
    }
    printf("\nGPU print end\n\n");
}

template <typename T1, typename T2, int LINEAR_BLOCK_SIZE = 256>
__device__ void
c_spline3d_reset_scratch_33x9x9data(volatile T1 shm_data[9][9][33], volatile T2 shm_errctrl[9][9][33], int radius)
{
    constexpr auto NUM_ITERS = 33 * 9 * 9 / LINEAR_BLOCK_SIZE + 1;  // 11 iterations
    // alternatively, reinterprete cast volatile T?[][][] to 1D
    for (auto i = 0; i < NUM_ITERS; i++) {
        auto _tix = i * LINEAR_BLOCK_SIZE + TIX;

        if (_tix < 33 * 9 * 9) {
            auto x = (_tix % 33);
            auto y = (_tix / 33) % 9;
            auto z = (_tix / 33) / 9;

            shm_data[z][y][x] = 0;
            /*****************************************************************************
             okay to use
             ******************************************************************************/
            if (x % 8 == 0 and y % 8 == 0 and z % 8 == 0) shm_errctrl[z][y][x] = radius;
            /*****************************************************************************
             alternatively
             ******************************************************************************/
            // shm_errctrl[z][y][x] = radius;
        }
    }
    __syncthreads();
}

template <typename T1, typename T2, int LINEAR_BLOCK_SIZE = 256>
__device__ void x_spline3d_reset_scratch_33x9x9data(
    volatile T1 shm_xdata[9][9][33],
    volatile T2 shm_errctrl[9][9][33],
    T1*         anchor,  // DIM3        anchor_dim3,  // unused
    STRIDE3     anchor_stride3)
{
    constexpr auto NUM_ITERS = 33 * 9 * 9 / LINEAR_BLOCK_SIZE + 1;  // 11 iterations
    // alternatively, reinterprete cast volatile T?[][][] to 1D
    for (auto i = 0; i < NUM_ITERS; i++) {
        auto _tix = i * LINEAR_BLOCK_SIZE + TIX;

        if (_tix < 33 * 9 * 9) {
            auto x = (_tix % 33);
            auto y = (_tix / 33) % 9;
            auto z = (_tix / 33) / 9;

            shm_errctrl[z][y][x] = 0;  // TODO explicitly handle zero-padding
            /*****************************************************************************
             okay to use
             ******************************************************************************/
            if (x % 8 == 0 and y % 8 == 0 and z % 8 == 0) {
                shm_xdata[z][y][x] = 0;

                auto aid = ((x / 8) + BIX * 4) +                 //
                           ((y / 8) + BIY) * anchor_stride3.y +  //
                           ((z / 8) + BIZ) * anchor_stride3.z;   //
                shm_xdata[z][y][x] = anchor[aid];
            }
            /*****************************************************************************
             alternatively
             ******************************************************************************/
            // shm_errctrl[z][y][x] = radius;
        }
    }
    __syncthreads();
}

template <typename Input, int LINEAR_BLOCK_SIZE = 256>
__device__ void
spline3d_global2shmem_33x9x9data(Input* data, volatile Input shm_data[9][9][33], DIM3 dim3d, STRIDE3 stride3d)
{
    constexpr auto TOTAL     = 33 * 9 * 9;
    constexpr auto NUM_ITERS = 33 * 9 * 9 / LINEAR_BLOCK_SIZE + 1;  // 11 iterations

    for (auto i = 0; i < NUM_ITERS; i++) {
        auto _tix = i * LINEAR_BLOCK_SIZE + TIX;

        if (_tix < TOTAL) {
            auto x   = (_tix % 33);
            auto y   = (_tix / 33) % 9;
            auto z   = (_tix / 33) / 9;
            auto gx  = (x + BIX * BLOCK32);
            auto gy  = (y + BIY * BLOCK8);
            auto gz  = (z + BIZ * BLOCK8);
            auto gid = gx + gy * stride3d.y + gz * stride3d.z;

            if (gx < dim3d.x and gy < dim3d.y and gz < dim3d.z) { shm_data[z][y][x] = data[gid]; }
        }
    }
    __syncthreads();
}

template <typename Output, int LINEAR_BLOCK_SIZE = 256>
__device__ void
spline3d_shmem2global_32x8x8data(volatile Output shm_data[9][9][33], Output* data, DIM3 dim3d, STRIDE3 stride3d)
{
    constexpr auto TOTAL     = 32 * 8 * 8;
    constexpr auto NUM_ITERS = TOTAL / LINEAR_BLOCK_SIZE + 1;  // 11 iterations

    for (auto i = 0; i < NUM_ITERS; i++) {
        auto _tix = i * LINEAR_BLOCK_SIZE + TIX;

        if (_tix < TOTAL) {
            auto x   = (_tix % 32);
            auto y   = (_tix / 32) % 8;
            auto z   = (_tix / 32) / 8;
            auto gx  = (x + BIX * BLOCK32);
            auto gy  = (y + BIY * BLOCK8);
            auto gz  = (z + BIZ * BLOCK8);
            auto gid = (x + BIX * BLOCK32) + (y + BIY * BLOCK8) * stride3d.y + (z + BIZ * BLOCK8) * stride3d.z;

            if (gx < dim3d.x and gy < dim3d.y and gz < dim3d.z) { data[gid] = shm_data[z][y][x]; }
        }
    }
    __syncthreads();
}

template <
    typename T1,
    typename T2,
    typename FP,
    typename LAMBDAX,
    typename LAMBDAY,
    typename LAMBDAZ,
    bool BLUE,
    bool YELLOW,
    bool HOLLOW,
    int  LINEAR_BLOCK_SIZE,
    int  BLOCK_DIMX,
    int  BLOCK_DIMY,
    bool COARSEN,
    int  BLOCK_DIMZ,
    bool BORDER_INCLUSIVE,
    bool WORKFLOW>
__forceinline__ __device__ void interpolate_stage(
    volatile T1 shm_data[9][9][33],
    volatile T2 shm_errctrl[9][9][33],
    LAMBDAX     xmap,
    LAMBDAY     ymap,
    LAMBDAZ     zmap,
    int         unit,
    FP          eb_r,
    FP          ebx2,
    int         radius)
{
    static_assert(BLOCK_DIMX * BLOCK_DIMY * (COARSEN ? 1 : BLOCK_DIMZ) <= LINEAR_BLOCK_SIZE, "block oversized");
    static_assert((BLUE or YELLOW or HOLLOW) == true, "must be one hot");
    static_assert((BLUE and YELLOW) == false, "must be only one hot (1)");
    static_assert((BLUE and YELLOW) == false, "must be only one hot (2)");
    static_assert((YELLOW and HOLLOW) == false, "must be only one hot (3)");

    auto run = [&](auto x, auto y, auto z) {
        if (xyz33x9x9_predicate<BORDER_INCLUSIVE>(x, y, z)) {
            T1 pred = 0;

            if CONSTEXPR (BLUE) {  //
                pred = (shm_data[z - unit][y][x] + shm_data[z + unit][y][x]) / 2;
            }
            if CONSTEXPR (YELLOW) {  //
                pred = (shm_data[z][y][x - unit] + shm_data[z][y][x + unit]) / 2;
            }
            if CONSTEXPR (HOLLOW) {  //
                pred = (shm_data[z][y - unit][x] + shm_data[z][y + unit][x]) / 2;
            }

            if CONSTEXPR (WORKFLOW == COMPRESS) {
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

    if CONSTEXPR (COARSEN) {
        constexpr auto TOTAL     = BLOCK_DIMX * BLOCK_DIMY * BLOCK_DIMZ;
        constexpr auto NUM_ITERS = TOTAL / LINEAR_BLOCK_SIZE + 1;

        for (auto i = 0; i < NUM_ITERS; i++) {
            auto _tix = i * LINEAR_BLOCK_SIZE + TIX;
            if (_tix < TOTAL) {
                auto itix = (_tix % BLOCK_DIMX);
                auto itiy = (_tix / BLOCK_DIMX) % BLOCK_DIMY;
                auto itiz = (_tix / BLOCK_DIMX) / BLOCK_DIMY;
                auto x    = xmap(itix, unit);
                auto y    = ymap(itiy, unit);
                auto z    = zmap(itiz, unit);
                run(x, y, z);
            }
        }
    }
    else {
        auto itix = (TIX % BLOCK_DIMX);
        auto itiy = (TIX / BLOCK_DIMX) % BLOCK_DIMY;
        auto itiz = (TIX / BLOCK_DIMX) / BLOCK_DIMY;
        auto x    = xmap(itix, unit);
        auto y    = ymap(itiy, unit);
        auto z    = zmap(itiz, unit);
        run(x, y, z);
    }
    __syncthreads();
}

}  // namespace

/********************************************************************************/

template <typename T1, typename T2, typename FP, int LINEAR_BLOCK_SIZE, bool WORKFLOW, bool ProbPredError>
__device__ void spline3d_layout2_interpolate(
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

    constexpr auto COARSEN          = true;
    constexpr auto NO_COARSEN       = false;
    constexpr auto BORDER_INCLUSIVE = true;
    constexpr auto BorderExclusive  = false;

    int unit = 4;

    // iteration 1
    interpolate_stage<
        T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
        true, false, false, LINEAR_BLOCK_SIZE, 5, 2, NO_COARSEN, 1, BORDER_INCLUSIVE, WORKFLOW>(
        shm_data, shm_errctrl, xblue, yblue, zblue, unit, eb_r, ebx2, radius);
    interpolate_stage<
        T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, 4, 2, NO_COARSEN, 3, BORDER_INCLUSIVE, WORKFLOW>(
        shm_data, shm_errctrl, xyellow, yyellow, zyellow, unit, eb_r, ebx2, radius);
    interpolate_stage<
        T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, 9, 1, NO_COARSEN, 3, BORDER_INCLUSIVE, WORKFLOW>(
        shm_data, shm_errctrl, xhollow, yhollow, zhollow, unit, eb_r, ebx2, radius);

    unit = 2;

    // iteration 2, TODO switch y-z order
    interpolate_stage<
        T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
        true, false, false, LINEAR_BLOCK_SIZE, 9, 3, NO_COARSEN, 2, BORDER_INCLUSIVE, WORKFLOW>(
        shm_data, shm_errctrl, xblue, yblue, zblue, unit, eb_r, ebx2, radius);
    interpolate_stage<
        T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, 8, 3, NO_COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
        shm_data, shm_errctrl, xyellow, yyellow, zyellow, unit, eb_r, ebx2, radius);
    interpolate_stage<
        T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, 17, 2, NO_COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
        shm_data, shm_errctrl, xhollow, yhollow, zhollow, unit, eb_r, ebx2, radius);

    unit = 1;

    // iteration 3
    interpolate_stage<
        T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
        true, false, false, LINEAR_BLOCK_SIZE, 17, 5, COARSEN, 4, BORDER_INCLUSIVE, WORKFLOW>(
        shm_data, shm_errctrl, xblue, yblue, zblue, unit, eb_r, ebx2, radius);
    interpolate_stage<
        T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, 16, 5, COARSEN, 9, BORDER_INCLUSIVE, WORKFLOW>(
        shm_data, shm_errctrl, xyellow, yyellow, zyellow, unit, eb_r, ebx2, radius);
    /******************************************************************************
     test only: last step inclusive
     ******************************************************************************/
    // interpolate_stage<
    //     T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
    //     false, false, true, LINEAR_BLOCK_SIZE, 33, 4, COARSEN, 9, BORDER_INCLUSIVE, WORKFLOW>(
    //     shm_data, shm_errctrl, xhollow, yhollow, zhollow, unit, eb_r, ebx2, radius);
    /******************************************************************************
     production
     ******************************************************************************/
    interpolate_stage<
        T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, 32, 4, COARSEN, 8, BorderExclusive, WORKFLOW>(
        shm_data, shm_errctrl, xhollow, yhollow, zhollow, unit, eb_r, ebx2, radius);

    /******************************************************************************
     test only: print a block
     ******************************************************************************/
    // if (TIX == 0 and BIX == 0 and BIY == 0 and BIZ == 0) { spline3d_print_block_from_GPU(shm_errctrl); }
    // if (TIX == 0 and BIX == 0 and BIY == 0 and BIZ == 0) { spline3d_print_block_from_GPU(shm_data); }
}

/********************************************************************************
 * host API/kernel
 ********************************************************************************/
template <
    typename DataIter,  //
    typename QuantIter,
    typename FP,
    int  LINEAR_BLOCK_SIZE,
    bool DELAY_POSTQUANT,
    bool ProbePredError>
__global__ void c_spline3d_infprecis_32x8x8data(
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
        // TODO
    }
    else {
        __shared__ struct {
            Data  data[9][9][33];
            Quant errctrl[9][9][33];
        } shmem;

        c_spline3d_reset_scratch_33x9x9data<Data, Quant, LINEAR_BLOCK_SIZE>(shmem.data, shmem.errctrl, radius);
        spline3d_global2shmem_33x9x9data<Data, LINEAR_BLOCK_SIZE>(data_first, shmem.data, data_dim3, data_stride3);
        spline3d_layout2_interpolate<Data, Quant, FP, LINEAR_BLOCK_SIZE, COMPRESS, false>(
            shmem.data, shmem.errctrl, eb_r, ebx2, radius);
        spline3d_shmem2global_32x8x8data<Quant, LINEAR_BLOCK_SIZE>(
            shmem.errctrl, errctrl_first, errctrl_dim3, errctrl_stride3);
    }
}

template <
    typename QuantIter,
    typename DataIter,
    typename FP,
    int  LINEAR_BLOCK_SIZE,
    bool DELAY_POSTQUANT>
__global__ void x_spline3d_infprecis_32x8x8data(
    QuantIter errctrl_first,    // input 1
    DIM3      errctrl_dim3,     //
    STRIDE3   errctrl_stride3,  //
    DataIter  anchor,           // input 2
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

    x_spline3d_reset_scratch_33x9x9data<Data, Quant, LINEAR_BLOCK_SIZE>(
        shmem.data, shmem.errctrl, anchor, anchor_stride3);
    spline3d_global2shmem_33x9x9data<Quant, LINEAR_BLOCK_SIZE>(
        errctrl_first, shmem.errctrl, errctrl_dim3, errctrl_stride3);
    spline3d_layout2_interpolate<Data, Quant, FP, LINEAR_BLOCK_SIZE, DECOMPRESS, false>(
        shmem.data, shmem.errctrl, eb_r, ebx2, radius);
    spline3d_shmem2global_32x8x8data<Data, LINEAR_BLOCK_SIZE>(shmem.data, data_first, data_dim3, data_stride3);
}

#endif
