/**
 * @file spline3.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-05-15
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_KERNEL_SPLINE3_CUH
#define CUSZ_KERNEL_SPLINE3_CUH

#include <stdint.h>
#include <stdio.h>
#include <type_traits>
#include "utils/cuda_err.cuh"
#include "utils/timer.h"

#define SPLINE3_COMPR true
#define SPLINE3_DECOMPR false

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
#define BDZ blockDim.

using DIM     = unsigned int;
using STRIDE  = unsigned int;
using DIM3    = dim3;
using STRIDE3 = dim3;

constexpr int BLOCK8  = 8;
constexpr int BLOCK32 = 32;

#define SHM_ERROR s_ectrl

namespace cusz {

/********************************************************************************
 * host API
 ********************************************************************************/

template <
    typename TITER,
    typename EITER,
    typename FP            = float,
    int  LINEAR_BLOCK_SIZE = 256,
    bool PROBE_PRED_ERROR  = false>
__global__ void c_spline3d_infprecis_32x8x8data(
    TITER   data,
    DIM3    data_size,
    STRIDE3 data_leap,
    EITER   ectrl,
    DIM3    ectrl_size,
    STRIDE3 ectrl_leap,
    TITER   anchor,
    STRIDE3 anchor_leap,
    FP      eb_r,
    FP      ebx2,
    int     radius,
    TITER   pred_error     = nullptr,
    TITER   compress_error = nullptr);

template <
    typename EITER,
    typename TITER,
    typename FP           = float,
    int LINEAR_BLOCK_SIZE = 256>
__global__ void x_spline3d_infprecis_32x8x8data(
    EITER   ectrl,        // input 1
    DIM3    ectrl_size,   //
    STRIDE3 ectrl_leap,   //
    TITER   anchor,       // input 2
    DIM3    anchor_size,  //
    STRIDE3 anchor_leap,  //
    TITER   data,         // output
    DIM3    data_size,    //
    STRIDE3 data_leap,    //
    FP      eb_r,
    FP      ebx2,
    int     radius);

namespace device_api {
/********************************************************************************
 * device API
 ********************************************************************************/
template <
    typename T1,
    typename T2,
    typename FP,
    int  LINEAR_BLOCK_SIZE,
    bool WORKFLOW         = SPLINE3_COMPR,
    bool PROBE_PRED_ERROR = false>
__device__ void
spline3d_layout2_interpolate(volatile T1 s_data[9][9][33], volatile T2 s_ectrl[9][9][33], FP eb_r, FP ebx2, int radius);
}  // namespace device_api

}  // namespace cusz

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
__device__ void
spline3d_print_block_from_GPU(T volatile a[9][9][33], int radius = 512, bool compress = true, bool print_ectrl = true)
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
                    T c = print_ectrl ? a[z][y][x] - radius : a[z][y][x];
                    if (compress) {
                        if (c == 0) { printf("%3c", '.'); }
                        else {
                            if (abs(c) >= 10) { printf("%3c", '*'); }
                            else {
                                if (print_ectrl) { printf("%3d", c); }
                                else {
                                    printf("%4.2f", c);
                                }
                            }
                        }
                    }
                    else {
                        if (print_ectrl) { printf("%3d", c); }
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
__device__ void c_reset_scratch_33x9x9data(volatile T1 s_data[9][9][33], volatile T2 s_ectrl[9][9][33], int radius)
{
    // alternatively, reinterprete cast volatile T?[][][] to 1D
    for (auto _tix = TIX; _tix < 33 * 9 * 9; _tix += LINEAR_BLOCK_SIZE) {
        auto x = (_tix % 33);
        auto y = (_tix / 33) % 9;
        auto z = (_tix / 33) / 9;

        s_data[z][y][x] = 0;
        /*****************************************************************************
         okay to use
         ******************************************************************************/
        if (x % 8 == 0 and y % 8 == 0 and z % 8 == 0) s_ectrl[z][y][x] = radius;
        /*****************************************************************************
         alternatively
         ******************************************************************************/
        // s_ectrl[z][y][x] = radius;
    }
    __syncthreads();
}

template <typename T1, int LINEAR_BLOCK_SIZE = 256>
__device__ void c_gather_anchor(T1* data, DIM3 data_size, STRIDE3 data_leap, T1* anchor, STRIDE3 anchor_leap)
{
    auto x = (TIX % 32) + BIX * 32;
    auto y = (TIX / 32) % 8 + BIY * 8;
    auto z = (TIX / 32) / 8 + BIZ * 8;

    bool pred1 = x % 8 == 0 and y % 8 == 0 and z % 8 == 0;
    bool pred2 = x < data_size.x and y < data_size.y and z < data_size.z;

    if (pred1 and pred2) {
        auto data_id      = x + y * data_leap.y + z * data_leap.z;
        auto anchor_id    = (x / 8) + (y / 8) * anchor_leap.y + (z / 8) * anchor_leap.z;
        anchor[anchor_id] = data[data_id];
    }
    __syncthreads();
}

/*
 * use shmem, erroneous
template <typename T1, int LINEAR_BLOCK_SIZE = 256>
__device__ void c_gather_anchor(volatile T1 s_data[9][9][33], T1* anchor, STRIDE3 anchor_leap)
{
    constexpr auto NUM_ITERS = 33 * 9 * 9 / LINEAR_BLOCK_SIZE + 1;  // 11 iterations
    for (auto i = 0; i < NUM_ITERS; i++) {
        auto _tix = i * LINEAR_BLOCK_SIZE + TIX;

        if (_tix < 33 * 9 * 9) {
            auto x = (_tix % 33);
            auto y = (_tix / 33) % 9;
            auto z = (_tix / 33) / 9;

            if (x % 8 == 0 and y % 8 == 0 and z % 8 == 0) {
                auto aid = ((x / 8) + BIX * 4) +             //
                           ((y / 8) + BIY) * anchor_leap.y +  //
                           ((z / 8) + BIZ) * anchor_leap.z;   //
                anchor[aid] = s_data[z][y][x];
            }
        }
    }
    __syncthreads();
}
*/

template <typename T1, typename T2 = T1, int LINEAR_BLOCK_SIZE = 256>
__device__ void x_reset_scratch_33x9x9data(
    volatile T1 s_xdata[9][9][33],
    volatile T2 s_ectrl[9][9][33],
    T1*         anchor,       //
    DIM3        anchor_size,  //
    STRIDE3     anchor_leap)
{
    for (auto _tix = TIX; _tix < 33 * 9 * 9; _tix += LINEAR_BLOCK_SIZE) {
        auto x = (_tix % 33);
        auto y = (_tix / 33) % 9;
        auto z = (_tix / 33) / 9;

        s_ectrl[z][y][x] = 0;  // TODO explicitly handle zero-padding
        /*****************************************************************************
         okay to use
         ******************************************************************************/
        if (x % 8 == 0 and y % 8 == 0 and z % 8 == 0) {
            s_xdata[z][y][x] = 0;

            auto ax = ((x / 8) + BIX * 4);
            auto ay = ((y / 8) + BIY);
            auto az = ((z / 8) + BIZ);

            if (ax < anchor_size.x and ay < anchor_size.y and az < anchor_size.z)
                s_xdata[z][y][x] = anchor[ax + ay * anchor_leap.y + az * anchor_leap.z];
        }
        /*****************************************************************************
         alternatively
         ******************************************************************************/
        // s_ectrl[z][y][x] = radius;
    }

    __syncthreads();
}

template <typename T1, typename T2, int LINEAR_BLOCK_SIZE = 256>
__device__ void global2shmem_33x9x9data(T1* data, DIM3 data_size, STRIDE3 data_leap, volatile T2 s_data[9][9][33])
{
    constexpr auto TOTAL = 33 * 9 * 9;

    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
        auto x   = (_tix % 33);
        auto y   = (_tix / 33) % 9;
        auto z   = (_tix / 33) / 9;
        auto gx  = (x + BIX * BLOCK32);
        auto gy  = (y + BIY * BLOCK8);
        auto gz  = (z + BIZ * BLOCK8);
        auto gid = gx + gy * data_leap.y + gz * data_leap.z;

        if (gx < data_size.x and gy < data_size.y and gz < data_size.z) s_data[z][y][x] = data[gid];
    }
    __syncthreads();
}

template <typename T1, typename T2, int LINEAR_BLOCK_SIZE = 256>
__device__ void
shmem2global_32x8x8data(volatile T1 s_data[9][9][33], T2* data, DIM3 data_size, STRIDE3 data_leap)
{
    constexpr auto TOTAL = 32 * 8 * 8;

    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
        auto x   = (_tix % 32);
        auto y   = (_tix / 32) % 8;
        auto z   = (_tix / 32) / 8;
        auto gx  = (x + BIX * BLOCK32);
        auto gy  = (y + BIY * BLOCK8);
        auto gz  = (z + BIZ * BLOCK8);
        auto gid = gx + gy * data_leap.y + gz * data_leap.z;

        if (gx < data_size.x and gy < data_size.y and gz < data_size.z) data[gid] = s_data[z][y][x];
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
    volatile T1 s_data[9][9][33],
    volatile T2 s_ectrl[9][9][33],
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
                pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
            }
            if CONSTEXPR (YELLOW) {  //
                pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
            }
            if CONSTEXPR (HOLLOW) {  //
                pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
            }

            if CONSTEXPR (WORKFLOW == SPLINE3_COMPR) {
                auto          err = s_data[z][y][x] - pred;
                decltype(err) code;
                // TODO unsafe, did not deal with the out-of-cap case
                {
                    code = fabs(err) * eb_r + 1;
                    code = err < 0 ? -code : code;
                    code = int(code / 2) + radius;
                }
                s_ectrl[z][y][x] = code;  // TODO double check if unsigned type works
                s_data[z][y][x]  = pred + (code - radius) * ebx2;
            }
            else {  // TODO == DECOMPRESSS and static_assert
                auto code       = s_ectrl[z][y][x];
                s_data[z][y][x] = pred + (code - radius) * ebx2;
            }
        }
    };
    // -------------------------------------------------------------------------------- //

    if CONSTEXPR (COARSEN) {
        constexpr auto TOTAL = BLOCK_DIMX * BLOCK_DIMY * BLOCK_DIMZ;
        for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
            auto itix = (_tix % BLOCK_DIMX);
            auto itiy = (_tix / BLOCK_DIMX) % BLOCK_DIMY;
            auto itiz = (_tix / BLOCK_DIMX) / BLOCK_DIMY;
            auto x    = xmap(itix, unit);
            auto y    = ymap(itiy, unit);
            auto z    = zmap(itiz, unit);
            run(x, y, z);
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

template <typename T1, typename T2, typename FP, int LINEAR_BLOCK_SIZE, bool WORKFLOW, bool PROBE_PRED_ERROR>
__device__ void cusz::device_api::spline3d_layout2_interpolate(
    volatile T1 s_data[9][9][33],
    volatile T2 s_ectrl[9][9][33],
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
    constexpr auto BORDER_EXCLUSIVE = false;

    int unit = 4;

    // iteration 1
    interpolate_stage<
        T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
        true, false, false, LINEAR_BLOCK_SIZE, 5, 2, NO_COARSEN, 1, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, xblue, yblue, zblue, unit, eb_r, ebx2, radius);
    interpolate_stage<
        T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, 4, 2, NO_COARSEN, 3, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, xyellow, yyellow, zyellow, unit, eb_r, ebx2, radius);
    interpolate_stage<
        T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, 9, 1, NO_COARSEN, 3, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, xhollow, yhollow, zhollow, unit, eb_r, ebx2, radius);

    unit = 2;

    // iteration 2, TODO switch y-z order
    interpolate_stage<
        T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
        true, false, false, LINEAR_BLOCK_SIZE, 9, 3, NO_COARSEN, 2, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, xblue, yblue, zblue, unit, eb_r, ebx2, radius);
    interpolate_stage<
        T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, 8, 3, NO_COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, xyellow, yyellow, zyellow, unit, eb_r, ebx2, radius);
    interpolate_stage<
        T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, 17, 2, NO_COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, xhollow, yhollow, zhollow, unit, eb_r, ebx2, radius);

    unit = 1;

    // iteration 3
    interpolate_stage<
        T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
        true, false, false, LINEAR_BLOCK_SIZE, 17, 5, COARSEN, 4, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, xblue, yblue, zblue, unit, eb_r, ebx2, radius);
    interpolate_stage<
        T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
        false, true, false, LINEAR_BLOCK_SIZE, 16, 5, COARSEN, 9, BORDER_INCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, xyellow, yyellow, zyellow, unit, eb_r, ebx2, radius);
    /******************************************************************************
     test only: last step inclusive
     ******************************************************************************/
    // interpolate_stage<
    //     T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
    //     false, false, true, LINEAR_BLOCK_SIZE, 33, 4, COARSEN, 9, BORDER_INCLUSIVE, WORKFLOW>(
    //     s_data, s_ectrl, xhollow, yhollow, zhollow, unit, eb_r, ebx2, radius);
    /******************************************************************************
     production
     ******************************************************************************/
    interpolate_stage<
        T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
        false, false, true, LINEAR_BLOCK_SIZE, 32, 4, COARSEN, 8, BORDER_EXCLUSIVE, WORKFLOW>(
        s_data, s_ectrl, xhollow, yhollow, zhollow, unit, eb_r, ebx2, radius);

    /******************************************************************************
     test only: print a block
     ******************************************************************************/
    // if (TIX == 0 and BIX == 0 and BIY == 0 and BIZ == 0) { spline3d_print_block_from_GPU(s_ectrl); }
    // if (TIX == 0 and BIX == 0 and BIY == 0 and BIZ == 0) { spline3d_print_block_from_GPU(s_data); }
}

/********************************************************************************
 * host API/kernel
 ********************************************************************************/

template <typename TITER, typename EITER, typename FP, int LINEAR_BLOCK_SIZE, bool PROBE_PRED_ERROR>
__global__ void cusz::c_spline3d_infprecis_32x8x8data(
    TITER   data,
    DIM3    data_size,
    STRIDE3 data_leap,
    EITER   ectrl,
    DIM3    ectrl_size,
    STRIDE3 ectrl_leap,
    TITER   anchor,
    STRIDE3 anchor_leap,
    FP      eb_r,
    FP      ebx2,
    int     radius,
    TITER   pred_error,
    TITER   compress_error)
{
    // compile time variables
    using T = typename std::remove_pointer<TITER>::type;
    using E = typename std::remove_pointer<EITER>::type;

    if CONSTEXPR (PROBE_PRED_ERROR) {
        // TODO
    }
    else {
        __shared__ struct {
            T data[9][9][33];
            T ectrl[9][9][33];
        } shmem;

        c_reset_scratch_33x9x9data<T, T, LINEAR_BLOCK_SIZE>(shmem.data, shmem.ectrl, radius);
        global2shmem_33x9x9data<T, T, LINEAR_BLOCK_SIZE>(data, data_size, data_leap, shmem.data);

        // version 1, use shmem, erroneous
        // c_gather_anchor<T>(shmem.data, anchor, anchor_leap);
        // version 2, use global mem, correct
        c_gather_anchor<T>(data, data_size, data_leap, anchor, anchor_leap);

        cusz::device_api::spline3d_layout2_interpolate<T, T, FP, LINEAR_BLOCK_SIZE, SPLINE3_COMPR, false>(
            shmem.data, shmem.ectrl, eb_r, ebx2, radius);
        shmem2global_32x8x8data<T, E, LINEAR_BLOCK_SIZE>(shmem.ectrl, ectrl, ectrl_size, ectrl_leap);
    }
}

template <
    typename EITER,
    typename TITER,
    typename FP,
    int LINEAR_BLOCK_SIZE>
__global__ void cusz::x_spline3d_infprecis_32x8x8data(
    EITER   ectrl,        // input 1
    DIM3    ectrl_size,   //
    STRIDE3 ectrl_leap,   //
    TITER   anchor,       // input 2
    DIM3    anchor_size,  //
    STRIDE3 anchor_leap,  //
    TITER   data,         // output
    DIM3    data_size,    //
    STRIDE3 data_leap,    //
    FP      eb_r,
    FP      ebx2,
    int     radius)
{
    // compile time variables
    using E = typename std::remove_pointer<EITER>::type;
    using T = typename std::remove_pointer<TITER>::type;

    __shared__ struct {
        T data[9][9][33];
        T ectrl[9][9][33];
    } shmem;

    x_reset_scratch_33x9x9data<T, T, LINEAR_BLOCK_SIZE>(shmem.data, shmem.ectrl, anchor, anchor_size, anchor_leap);
    global2shmem_33x9x9data<E, T, LINEAR_BLOCK_SIZE>(ectrl, ectrl_size, ectrl_leap, shmem.ectrl);
    cusz::device_api::spline3d_layout2_interpolate<T, T, FP, LINEAR_BLOCK_SIZE, SPLINE3_DECOMPR, false>(
        shmem.data, shmem.ectrl, eb_r, ebx2, radius);
    shmem2global_32x8x8data<T, T, LINEAR_BLOCK_SIZE>(shmem.data, data, data_size, data_leap);
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

template <typename T, typename E, typename FP, bool NO_R_SEPARATE>
void launch_construct_Spline3(
    T*           data,
    dim3 const   len3,
    T*           anchor,
    dim3 const   an_len3,
    E*           ectrl,
    dim3 const   ec_len3,
    double const eb,
    int const    radius,
    float&       time_elapsed,
    cudaStream_t stream)
{
    auto divide3 = [](dim3 len, dim3 sublen) {
        return dim3(
            (len.x - 1) / sublen.x + 1,  //
            (len.y - 1) / sublen.y + 1,  //
            (len.z - 1) / sublen.z + 1);
    };

    auto ndim = [&]() {
        if (len3.z == 1 and len3.y == 1)
            return 1;
        else if (len3.z == 1 and len3.y != 1)
            return 2;
        else
            return 3;
    };

    constexpr auto SUBLEN_3D = dim3(32, 8, 8);
    constexpr auto SEQ_3D    = dim3(1, 8, 1);
    constexpr auto BLOCK_3D  = dim3(256, 1, 1);
    auto           GRID_3D   = divide3(len3, SUBLEN_3D);

    {
        constexpr auto SUBLEN_TOTAL = SUBLEN_3D.x * SUBLEN_3D.y * SUBLEN_3D.z;
        constexpr auto SEQ_TOTAL    = SEQ_3D.x * SEQ_3D.y * SEQ_3D.z;
        constexpr auto BLOCK_TOTAL  = BLOCK_3D.x * BLOCK_3D.y * BLOCK_3D.z;

        // static_assert(SUBLEN_TOTAL / SEQ_TOTAL == BLOCK_TOTAL, "parallelism does not match!");
        if (SUBLEN_TOTAL / SEQ_TOTAL != BLOCK_TOTAL) throw std::runtime_error("parallelism does not match!");
    }

    ////////////////////////////////////////

    auto ebx2     = eb * 2;
    auto eb_r     = 1 / eb;
    auto leap3    = dim3(1, len3.x, len3.x * len3.y);
    auto ec_leap3 = dim3(1, ec_len3.x, ec_len3.x * ec_len3.y);
    auto an_leap3 = dim3(1, an_len3.x, an_len3.x * an_len3.y);

    CREATE_CUDAEVENT_PAIR;
    START_CUDAEVENT_RECORDING(stream);

    auto d = ndim();

    if (d == 1) {  //
        throw std::runtime_error("Spline1 not implemented");
    }
    else if (d == 2) {
        throw std::runtime_error("Spline2 not implemented");
    }
    else if (d == 3) {
        cusz::c_spline3d_infprecis_32x8x8data<T*, E*, float, 256, false>  //
            <<<GRID_3D, BLOCK_3D, 0, stream>>>                            //
            (data, len3, leap3,                                           //
             ectrl, ec_len3, ec_leap3,                                    //
             anchor, an_leap3,                                            //
             eb_r, ebx2, radius);
    }

    STOP_CUDAEVENT_RECORDING(stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    TIME_ELAPSED_CUDAEVENT(&time_elapsed);

    DESTROY_CUDAEVENT_PAIR;
}

template <typename T, typename E, typename FP>
void launch_reconstruct_Spline3(
    T*           xdata,
    dim3 const   len3,
    T*           anchor,
    dim3 const   an_len3,
    E*           ectrl,
    dim3 const   ec_len3,
    double const eb,
    int const    radius,
    float&       time_elapsed,
    cudaStream_t stream)
{
    auto divide3 = [](dim3 len, dim3 sublen) {
        return dim3(
            (len.x - 1) / sublen.x + 1,  //
            (len.y - 1) / sublen.y + 1,  //
            (len.z - 1) / sublen.z + 1);
    };

    /*
    auto ndim = [&]() {
        if (len3.z == 1 and len3.y == 1)
            return 1;
        else if (len3.z == 1 and len3.y != 1)
            return 2;
        else
            return 3;
    };
     */

    constexpr auto SUBLEN_3D = dim3(32, 8, 8);
    constexpr auto SEQ_3D    = dim3(1, 8, 1);
    constexpr auto BLOCK_3D  = dim3(256, 1, 1);
    auto           GRID_3D   = divide3(len3, SUBLEN_3D);

    {
        constexpr auto SUBLEN_TOTAL = SUBLEN_3D.x * SUBLEN_3D.y * SUBLEN_3D.z;
        constexpr auto SEQ_TOTAL    = SEQ_3D.x * SEQ_3D.y * SEQ_3D.z;
        constexpr auto BLOCK_TOTAL  = BLOCK_3D.x * BLOCK_3D.y * BLOCK_3D.z;

        // static_assert(SUBLEN_TOTAL / SEQ_TOTAL == BLOCK_TOTAL, "parallelism does not match!");
        if (SUBLEN_TOTAL / SEQ_TOTAL != BLOCK_TOTAL) throw std::runtime_error("parallelism does not match!");
    }

    ////////////////////////////////////////

    auto ebx2     = eb * 2;
    auto eb_r     = 1 / eb;
    auto leap3    = dim3(1, len3.x, len3.x * len3.y);
    auto ec_leap3 = dim3(1, ec_len3.x, ec_len3.x * ec_len3.y);
    auto an_leap3 = dim3(1, an_len3.x, an_len3.x * an_len3.y);

    CREATE_CUDAEVENT_PAIR;
    START_CUDAEVENT_RECORDING(stream);

    cusz::x_spline3d_infprecis_32x8x8data<E*, T*, float, 256>  //
        <<<GRID_3D, BLOCK_3D, 0, stream>>>                     //
        (ectrl, ec_len3, ec_leap3,                             //
         anchor, an_len3, an_leap3,                            //
         xdata, len3, leap3,                                   //
         eb_r, ebx2, radius);

    STOP_CUDAEVENT_RECORDING(stream);

    CHECK_CUDA(cudaStreamSynchronize(stream));

    TIME_ELAPSED_CUDAEVENT(&time_elapsed);
    DESTROY_CUDAEVENT_PAIR;
}

#endif
