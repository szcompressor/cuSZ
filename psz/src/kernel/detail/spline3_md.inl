 /**
 * @file spline3.inl
 * @author Jinyang Liu, Shixun Wu, Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-05-15
 *
 * (copyright to be updated)
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_KERNEL_SPLINE3_CUH
#define CUSZ_KERNEL_SPLINE3_CUH

#include <stdint.h>
#include <stdio.h>
#include <type_traits>
#include <tuple>
#include "cusz/type.h"
#include "utils/err.hh"
#include "utils/timer.hh"

#define SPLINE3_COMPR true
#define SPLINE3_DECOMPR false

#define SPLINE3_PRED_ATT true
#define SPLINE3_AB_ATT false

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
#define GDX gridDim.x
#define GDY gridDim.y
#define GDZ gridDim.z

using DIM     = u4;
using STRIDE  = u4;
using DIM3    = dim3;
using STRIDE3 = dim3;
#define BLOCK_DIM_SIZE 384

constexpr int BLOCK16 = 16;
constexpr int BLOCK17 = 17;
constexpr int DEFAULT_LINEAR_BLOCK_SIZE = BLOCK_DIM_SIZE;

#define SHM_ERROR s_ectrl

namespace cusz {

/********************************************************************************
 * host API
 ********************************************************************************/
template <typename TITER, int SPLINE_DIM,
int PROFILE_BLOCK_SIZE_X,
int PROFILE_BLOCK_SIZE_Y,
int PROFILE_BLOCK_SIZE_Z,
int PROFILE_NUM_BLOCK_X,
int PROFILE_NUM_BLOCK_Y,
int PROFILE_NUM_BLOCK_Z,
int  LINEAR_BLOCK_SIZE>
__global__ void c_spline_profiling_data(
    TITER   data,
    DIM3    data_size,
    STRIDE3 data_leap,
    TITER errors);

template <typename TITER, int SPLINE_DIM,
int PROFILE_NUM_BLOCK_X,
int PROFILE_NUM_BLOCK_Y,
int PROFILE_NUM_BLOCK_Z,
int  LINEAR_BLOCK_SIZE>
__global__ void c_spline_profiling_data_2(
    TITER   data,
    DIM3    data_size,
    STRIDE3 data_leap,
    TITER errors);

template <
    typename TITER,
    typename EITER,
    typename FP            = float,
    int LEVEL = 4,
    int SPLINE_DIM = 2,
    int AnchorBlockSizeX = 8, int AnchorBlockSizeY = 8,
    int AnchorBlockSizeZ = 1,
    int numAnchorBlockX = 4,  // Number of Anchor blocks along X
    int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
    int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
    int  LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE, 
    typename CompactVal = TITER,
    typename CompactIdx = uint32_t*,
    typename CompactNum = uint32_t*>
__global__ void c_spline_infprecis_data(
    TITER   data,
    DIM3    data_size,
    STRIDE3 data_leap,
    EITER   ectrl,
    DIM3    ectrl_size,
    STRIDE3 ectrl_leap,
    TITER   anchor,
    STRIDE3 anchor_leap,
    CompactVal cval,
    CompactIdx cidx,
    CompactNum cn,
    FP      eb_r,
    FP      ebx2,
    int     radius,
    INTERPOLATION_PARAMS intp_param,
    TITER errors);

template <
    typename EITER,
    typename TITER,
    typename FP           = float,
    int LEVEL = 4,
    int SPLINE_DIM = 2,
    int AnchorBlockSizeX = 8, int AnchorBlockSizeY = 8,
    int AnchorBlockSizeZ = 1,
    int numAnchorBlockX = 4,  // Number of Anchor blocks along X
    int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
    int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
    int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__global__ void x_spline_infprecis_data(
    EITER   ectrl,        // input 1
    DIM3    ectrl_size,   //
    STRIDE3 ectrl_leap,   //
    TITER   anchor,       // input 2
    DIM3    anchor_size,  //
    STRIDE3 anchor_leap,  //
    TITER   data,         // output
    DIM3    data_size,    //
    STRIDE3 data_leap,    //
    TITER   outlier_tmp,
    FP      eb_r,
    FP      ebx2,
    int     radius,
    INTERPOLATION_PARAMS intp_param);

template <typename TITER>
__global__ void reset_errors(TITER errors);

template <typename TITER, typename FP, 
int LEVEL = 4,
int SPLINE_DIM = 2,
int AnchorBlockSizeX = 8, int AnchorBlockSizeY = 8,
int AnchorBlockSizeZ = 1,
int numAnchorBlockX = 4,  // Number of Anchor blocks along X
int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__global__ void pa_spline_infprecis_data(
    TITER   data,
    DIM3    data_size,
    STRIDE3 data_leap,
    DIM3 sample_starts,
    DIM3 sample_block_grid_sizes,
    DIM3 sample_strides,
    FP      eb_r,
    FP      ebx2,
    INTERPOLATION_PARAMS intp_param,
    TITER errors,
    bool workflow = SPLINE3_PRED_ATT
    );


namespace device_api {
/********************************************************************************
 * device API
 ********************************************************************************/
    template <typename T, 
    int SPLINE_DIM = 3,
int PROFILE_BLOCK_SIZE_X = 4,
int PROFILE_BLOCK_SIZE_Y = 4,
int PROFILE_BLOCK_SIZE_Z = 4,
int PROFILE_NUM_BLOCK_X = 4,
int PROFILE_NUM_BLOCK_Y = 4,
int PROFILE_NUM_BLOCK_Z = 4,
    int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void auto_tuning(
    volatile T s_data[PROFILE_BLOCK_SIZE_Z * PROFILE_NUM_BLOCK_Z]       
                    [PROFILE_BLOCK_SIZE_Y * PROFILE_NUM_BLOCK_Y][PROFILE_BLOCK_SIZE_X * PROFILE_NUM_BLOCK_X],  volatile T local_errs[6], DIM3  data_size, volatile T* count);

 template <typename T, 
 int SPLINE_DIM = 3,
int PROFILE_NUM_BLOCK_X = 4,
int PROFILE_NUM_BLOCK_Y = 4,
int PROFILE_NUM_BLOCK_Z = 4,
 int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void auto_tuning_2(
    volatile T s_data[PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z], volatile T s_nx[PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z][4], volatile T s_ny[PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z][4], volatile T s_nz[PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z][4],  volatile T local_errs[6], DIM3  data_size, volatile T* count);

template <
    typename T1,
    typename T2,
    typename FP,
    int LEVEL, 
    int SPLINE_DIM = 2,
int AnchorBlockSizeX = 8, int AnchorBlockSizeY = 8,
int AnchorBlockSizeZ = 1,
int numAnchorBlockX = 4,  // Number of Anchor blocks along X
int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE,
    bool WORKFLOW         = SPLINE3_COMPR,
    bool PROBE_PRED_ERROR = false>
__device__ void
spline_layout_interpolate(
    volatile T1 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]       
                    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)][AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)], 
    volatile T2 s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]       
                    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)][AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],  DIM3  data_size,FP eb_r, FP ebx2, int radius, INTERPOLATION_PARAMS intp_param);

template <typename T, typename FP, int LEVEL, int SPLINE_DIM,
int AnchorBlockSizeX, int AnchorBlockSizeY,
int AnchorBlockSizeZ,
int numAnchorBlockX,  // Number of Anchor blocks along X
int numAnchorBlockY,  // Number of Anchor blocks along Y
int numAnchorBlockZ,  // Number of Anchor blocks along Z
int LINEAR_BLOCK_SIZE, bool WORKFLOW>
__device__ void spline_layout_interpolate_att(
    volatile T s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]       
                    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)][AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
     DIM3    data_size,
    DIM3 global_starts,FP eb_r, FP ebx2,uint8_t level,INTERPOLATION_PARAMS intp_param,volatile T *error);


}  // namespace device_api




}  // namespace cusz

/********************************************************************************
 * helper function
 ********************************************************************************/

namespace {

template <int SPLINE_DIM, 
int AnchorBlockSizeX,
int AnchorBlockSizeY, 
int AnchorBlockSizeZ,
int numAnchorBlockX,  // Number of Anchor blocks along X
int numAnchorBlockY,  // Number of Anchor blocks along Y
int numAnchorBlockZ,  // Number of Anchor blocks along Z
bool INCLUSIVE = true>
__forceinline__ __device__ bool xyz_predicate(unsigned int x, unsigned int y, unsigned int z,const DIM3 &data_size)
{
    if CONSTEXPR (INCLUSIVE) {  //

        
            return (x <= (AnchorBlockSizeX * numAnchorBlockX) and y <= (AnchorBlockSizeY * numAnchorBlockY) and z <= (AnchorBlockSizeZ * numAnchorBlockZ)) and BIX * (AnchorBlockSizeX * numAnchorBlockX) + x < data_size.x and BIY *  (AnchorBlockSizeY * numAnchorBlockY) + y < data_size.y and BIZ * (AnchorBlockSizeZ * numAnchorBlockZ) + z < data_size.z;
    }
    else {
        return x < (AnchorBlockSizeX * numAnchorBlockX) + (BIX == GDX - 1) * (SPLINE_DIM <= 1) and y < (AnchorBlockSizeY * numAnchorBlockY) + (BIY == GDY - 1) * (SPLINE_DIM <= 2) and z < (AnchorBlockSizeZ * numAnchorBlockZ) + (BIZ == GDZ - 1) * (SPLINE_DIM <= 3) and BIX * (AnchorBlockSizeX * numAnchorBlockX) + x < data_size.x and BIY * (AnchorBlockSizeY * numAnchorBlockY) + y < data_size.y and BIZ * (AnchorBlockSizeZ * numAnchorBlockZ) + z < data_size.z;
    }
}

template <int SPLINE_DIM,
int AnchorBlockSizeX,
int AnchorBlockSizeY,
int AnchorBlockSizeZ,
int numAnchorBlockX, 
int numAnchorBlockY, 
int numAnchorBlockZ, 
bool INCLUSIVE = true>
__forceinline__ __device__ bool xyz_predicate_att(unsigned int x, unsigned int y, unsigned int z,const DIM3 &data_size, const DIM3 & global_starts)
{
    if CONSTEXPR (INCLUSIVE) {
            return (x <= (AnchorBlockSizeX * numAnchorBlockX) and y <= (AnchorBlockSizeY * numAnchorBlockY) and z <= (AnchorBlockSizeZ * numAnchorBlockZ)) and global_starts.x + x < data_size.x and global_starts.y + y < data_size.y and global_starts.z + z < data_size.z;
    }
    else {
        return x < (AnchorBlockSizeX * numAnchorBlockX) + (BIX == GDX - 1) and y < (AnchorBlockSizeY * numAnchorBlockY) + (BIY == GDY - 1) and z < (AnchorBlockSizeZ * numAnchorBlockZ) + (BIZ == GDZ - 1) and global_starts.x + x < data_size.x and global_starts.y + y < data_size.y and global_starts.z + z < data_size.z;
    }
}

template <typename T1, typename T2,
int SPLINE_DIM, 
int AnchorBlockSizeX, 
int AnchorBlockSizeY,
int AnchorBlockSizeZ,
int numAnchorBlockX,  // Number of Anchor blocks along X
int numAnchorBlockY,  // Number of Anchor blocks along Y
int numAnchorBlockZ,  // Number of Anchor blocks along Z 
int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void c_reset_scratch_data(
    volatile T1 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                       [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                       [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    volatile T2 s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                        [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                        [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    int radius)
{
    for (auto _tix = TIX; _tix < (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)) *
            (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)) * (AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3));
            _tix += LINEAR_BLOCK_SIZE) {
        auto x = (_tix % (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)));
        auto y = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) %
                    (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
        auto z = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) /
                    (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));

        s_data[z][y][x] = 0;
        if (x % AnchorBlockSizeX == 0 and y % AnchorBlockSizeY == 0 and
            z % AnchorBlockSizeZ == 0)
          s_ectrl[z][y][x] = radius;
    }
    __syncthreads();
}


template <typename T, 
int SPLINE_DIM = 3,
int PROFILE_BLOCK_SIZE_X = 4,
int PROFILE_BLOCK_SIZE_Y = 4,
int PROFILE_BLOCK_SIZE_Z = 4,
int PROFILE_NUM_BLOCK_X = 4,
int PROFILE_NUM_BLOCK_Y = 4,
int PROFILE_NUM_BLOCK_Z = 4,
int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void c_reset_scratch_profiling_data(
    volatile T s_data[PROFILE_BLOCK_SIZE_Z * PROFILE_NUM_BLOCK_Z][PROFILE_BLOCK_SIZE_Y * PROFILE_NUM_BLOCK_Y][PROFILE_BLOCK_SIZE_X * PROFILE_NUM_BLOCK_X], T default_value)
{
    auto x_size = PROFILE_BLOCK_SIZE_X * PROFILE_NUM_BLOCK_X;
    auto y_size = PROFILE_BLOCK_SIZE_Y * PROFILE_NUM_BLOCK_Y;
    auto z_size = PROFILE_BLOCK_SIZE_Z * PROFILE_NUM_BLOCK_Z;
    for (auto _tix = TIX; _tix < x_size * y_size * z_size; _tix += LINEAR_BLOCK_SIZE) {
        auto x = (_tix % x_size);
        auto y = (_tix / x_size) % y_size;
        auto z = (_tix / x_size) / y_size;
        s_data[z][y][x] = default_value;

    }
}

template <typename T,
int SPLINE_DIM = 3,
int PROFILE_NUM_BLOCK_X = 4,
int PROFILE_NUM_BLOCK_Y = 4,
int PROFILE_NUM_BLOCK_Z = 4, 
int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void c_reset_scratch_profiling_data_2(volatile T s_data[PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z], T nx[PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z][4], T ny[PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z][4], T nz[PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z][4],T default_value)
{
    for (auto _tix = TIX; _tix < PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z * 4; _tix += LINEAR_BLOCK_SIZE) {
        auto offset = (_tix % 4);
        auto idx = _tix / 4;
        nx[idx][offset] = ny[idx][offset] = nz[idx][offset] = default_value;
        // s_data[TIX] = default_value;
        s_data[idx] = default_value;

    }
}


template <typename T1, 
int AnchorBlockSizeX, int AnchorBlockSizeY,
     int AnchorBlockSizeZ,
     int numAnchorBlockX,  // Number of Anchor blocks along X
     int numAnchorBlockY,  // Number of Anchor blocks along Y
     int numAnchorBlockZ,  // Number of Anchor blocks along Z
     int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void c_gather_anchor(T1* data, DIM3 data_size, STRIDE3 data_leap, T1* anchor, STRIDE3 anchor_leap)
{
    auto ax = BIX ;//1 is block16 by anchor stride
    auto ay = BIY ;
    auto az = BIZ ;
    // 2d bug may be here! 
    auto x = (AnchorBlockSizeX * numAnchorBlockX) * ax;
    auto y = (AnchorBlockSizeY * numAnchorBlockY) * ay;
    auto z = (AnchorBlockSizeZ * numAnchorBlockZ) * az;

    bool pred1 = TIX < 1;//1 is num of anchor
    bool pred2 = x < data_size.x and y < data_size.y and z < data_size.z;

    if (pred1 and pred2) {
        auto data_id      = x + y * data_leap.y + z * data_leap.z;
        auto anchor_id    = ax + ay * anchor_leap.y + az * anchor_leap.z;
        anchor[anchor_id] = data[data_id];
    }
    __syncthreads();
}



template <typename T1, typename T2 = T1,
int SPLINE_DIM = 2, int AnchorBlockSizeX = 8,
     int AnchorBlockSizeY = 8, int AnchorBlockSizeZ = 8,
     int numAnchorBlockX = 4,  // Number of Anchor blocks along X
     int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
     int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
      int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void x_reset_scratch_data(
    volatile T1 s_xdata[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                        [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                        [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    volatile T2 s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                    [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    T1*         anchor,       //
    DIM3        anchor_size,  //
    STRIDE3     anchor_leap)
{
    for (auto _tix = TIX; _tix <  (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)) * (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)) * (AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)); _tix += LINEAR_BLOCK_SIZE) {
        auto x = (_tix % (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)));
        auto y = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) %
                 (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
        auto z = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) /
                 (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));

        s_ectrl[z][y][x] = 0;  // TODO explicitly handle zero-padding
        /*****************************************************************************
         okay to use
         ******************************************************************************/
        if (x % AnchorBlockSizeX == 0 and y % AnchorBlockSizeY == 0 and
            z % AnchorBlockSizeZ == 0) {
            s_xdata[z][y][x] = 0;

            auto ax = ((x / AnchorBlockSizeX) + BIX * numAnchorBlockX);
            auto ay = ((y / AnchorBlockSizeY) + BIY * numAnchorBlockY);
            auto az = ((z / AnchorBlockSizeZ) + BIZ * numAnchorBlockZ);

            if (ax < anchor_size.x and ay < anchor_size.y and az < anchor_size.z)
                s_xdata[z][y][x] = anchor[ax + ay * anchor_leap.y + az * anchor_leap.z];

        }

    }

    __syncthreads();
}

template <typename T1, typename T2, int SPLINE_DIM = 2, int AnchorBlockSizeX = 8,
int AnchorBlockSizeY = 8, int AnchorBlockSizeZ = 8,
int numAnchorBlockX = 4,  // Number of Anchor blocks along X
int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void global2shmem_data(T1* data, DIM3 data_size, STRIDE3 data_leap,
    volatile T2 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                    [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)])
{
    constexpr auto TOTAL = (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)) *
                        (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)) * 
                        (AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3));

    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
        auto x = (_tix % (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)));
        auto y = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) %
                 (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
        auto z = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) /
                 (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
        auto gx  = (x + BIX * (AnchorBlockSizeX * numAnchorBlockX));
        auto gy  = (y + BIY * (AnchorBlockSizeY * numAnchorBlockY));
        auto gz  = (z + BIZ * (AnchorBlockSizeZ * numAnchorBlockZ));
        auto gid = gx + gy * data_leap.y + gz * data_leap.z;

        if (gx < data_size.x and gy < data_size.y and gz < data_size.z) s_data[z][y][x] = data[gid];

    }
    __syncthreads();
}

template <typename T1, typename T2, 
int SPLINE_DIM = 3,
int PROFILE_BLOCK_SIZE_X = 4,
int PROFILE_BLOCK_SIZE_Y = 4,
int PROFILE_BLOCK_SIZE_Z = 4,
int PROFILE_NUM_BLOCK_X = 4,
int PROFILE_NUM_BLOCK_Y = 4,
int PROFILE_NUM_BLOCK_Z = 4,
int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void global2shmem_profiling_data(T1* data, DIM3 data_size, STRIDE3 data_leap, volatile T2 s_data[PROFILE_BLOCK_SIZE_Z * PROFILE_NUM_BLOCK_Z][PROFILE_BLOCK_SIZE_Y * PROFILE_NUM_BLOCK_Y][PROFILE_BLOCK_SIZE_X * PROFILE_NUM_BLOCK_X])
{
    constexpr auto x_size = PROFILE_BLOCK_SIZE_X * PROFILE_NUM_BLOCK_X;
    constexpr auto y_size = PROFILE_BLOCK_SIZE_Y * PROFILE_NUM_BLOCK_Y;
    constexpr auto z_size = PROFILE_BLOCK_SIZE_Z * PROFILE_NUM_BLOCK_Z;
    constexpr auto TOTAL = x_size * y_size * z_size;

    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
        auto x   = (_tix % x_size);
        auto y   = (_tix / x_size) % y_size;
        auto z   = (_tix / x_size) / y_size;
        auto gx_1 = x / PROFILE_BLOCK_SIZE_X;
        auto gx_2 = x % PROFILE_BLOCK_SIZE_X;
        auto gy_1 = y / PROFILE_BLOCK_SIZE_Y;
        auto gy_2 = y % PROFILE_BLOCK_SIZE_Y;
        auto gz_1 = z / PROFILE_BLOCK_SIZE_Z;
        auto gz_2 = z % PROFILE_BLOCK_SIZE_Z;
        auto gx = (data_size.x / PROFILE_NUM_BLOCK_X) * gx_1 + gx_2;
        auto gy = (data_size.y / PROFILE_NUM_BLOCK_Y) * gy_1 + gy_2;
        auto gz = (data_size.z / PROFILE_NUM_BLOCK_Z) * gz_1 + gz_2;

        auto gid = gx + gy * data_leap.y + gz * data_leap.z;

        if (gx < data_size.x and gy < data_size.y and gz < data_size.z) s_data[z][y][x] = data[gid];

    }
    __syncthreads();
}

template <typename T1, typename T2,
int SPLINE_DIM = 3,
int PROFILE_NUM_BLOCK_X = 4,
int PROFILE_NUM_BLOCK_Y = 4,
int PROFILE_NUM_BLOCK_Z = 4,
int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void global2shmem_profiling_data_2(T1* data, DIM3 data_size, STRIDE3 data_leap, volatile T2 s_data[PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z], volatile T2 s_nx[PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z][4], volatile T2 s_ny[PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z][4], volatile T2 s_nz[PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z][4])
{
    constexpr auto TOTAL = PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z * 4;
    int factors[4] = {-3, -1, 1, 3};
    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
        auto offset   = (_tix % 4);
        auto idx = _tix / 4;
        auto x = idx % PROFILE_NUM_BLOCK_X;
        auto y   = (idx / PROFILE_NUM_BLOCK_X) % PROFILE_NUM_BLOCK_Y;
        auto z   = (idx / PROFILE_NUM_BLOCK_X) / PROFILE_NUM_BLOCK_Y;
        auto gx=(data_size.x / PROFILE_NUM_BLOCK_X) * x + data_size.x / (PROFILE_NUM_BLOCK_X * 2);
        auto gy=(data_size.y / PROFILE_NUM_BLOCK_Y) * y + data_size.y / (PROFILE_NUM_BLOCK_Y * 2);
        auto gz=(data_size.z / PROFILE_NUM_BLOCK_Z) * z + data_size.z / (PROFILE_NUM_BLOCK_Z * 2);

        auto gid = gx + gy * data_leap.y + gz * data_leap.z;

        if CONSTEXPR (SPLINE_DIM == 3){
            if (gx >= 3 and gy >= 3 and gz >= 3 and gx + 3 < data_size.x and gy + 3 < data_size.y and gz + 3 < data_size.z) {
                s_data[idx] = data[gid];
                auto factor=factors[offset];
                s_nx[idx][offset]=data[gid+factor];
                s_ny[idx][offset]=data[gid+factor*data_leap.y];
                s_nz[idx][offset]=data[gid+factor*data_leap.z];
            }
        }

        if CONSTEXPR (SPLINE_DIM == 2){
            if (gx >= 3 and gy >= 3 and gx + 3 < data_size.x and gy + 3 < data_size.y) {
                s_data[idx] = data[gid];
                auto factor=factors[offset];
                s_nx[idx][offset]=data[gid+factor];
                s_ny[idx][offset]=data[gid+factor*data_leap.y];
            }
        }

    }
    __syncthreads();
}

template <typename T = float, typename E = u4, int LEVEL = 4,
int SPLINE_DIM = 2, int AnchorBlockSizeX = 8,
int AnchorBlockSizeY = 8, int AnchorBlockSizeZ = 8,
int numAnchorBlockX = 4,  // Number of Anchor blocks along X
int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void global2shmem_fuse(E* ectrl, dim3 ectrl_size, dim3 ectrl_leap, T* scattered_outlier, 
    volatile T s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
    [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    volatile size_t grid_leaps[LEVEL + 1][2],volatile size_t prefix_nums[LEVEL + 1])
{
    
    constexpr auto TOTAL = (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)) *
    (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)) *
    (AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3));

    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
        auto x   = (_tix % (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)));
        auto y   = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) % (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
        auto z   = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) / (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
        auto gx  = (x + BIX * (AnchorBlockSizeX * numAnchorBlockX));
        auto gy  = (y + BIY * (AnchorBlockSizeY * numAnchorBlockY));
        auto gz  = (z + BIZ * (AnchorBlockSizeZ * numAnchorBlockZ));
        if (gx < ectrl_size.x and gy < ectrl_size.y and gz < ectrl_size.z) {
            //todo: pre-compute the leaps and their halves

            int level = 0;
            auto data_gid = gx + gy * ectrl_leap.y + gz * ectrl_leap.z;
            while(gx % 2 == 0 and gy % 2 == 0 and gz % 2 == 0 and level < LEVEL){
                gx = gx >> 1;
                gy = gy >> 1;
                gz = gz >> 1;
                level++;
            }
            auto gid = gx + gy*grid_leaps[level][0] + gz*grid_leaps[level][1];

            if(level < LEVEL){//non-anchor
                gid += prefix_nums[level] - ((gz + 1) >> 1) * grid_leaps[level + 1][1] - (gz % 2 == 0) * ((gy + 1) >> 1) * grid_leaps[level + 1][0] - (gz % 2 == 0 && gy % 2 == 0) * ((gx + 1) >> 1);
            }

            s_ectrl[z][y][x] = static_cast<T>(ectrl[gid]) + scattered_outlier[data_gid];
        }
    }
    __syncthreads();
}

// dram_outlier should be the same in type with shared memory buf
template <typename T1, typename T2,
int SPLINE_DIM, int AnchorBlockSizeX,
int AnchorBlockSizeY, int AnchorBlockSizeZ,
int numAnchorBlockX,
int numAnchorBlockY,
int numAnchorBlockZ,
int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void
shmem2global_data(volatile T1 s_buf[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
[AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
[AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)], T2* dram_buf, DIM3 buf_size, STRIDE3 buf_leap)
{
    auto x_size = AnchorBlockSizeX * numAnchorBlockX + (BIX == GDX - 1) * (SPLINE_DIM >= 1);
    auto y_size = AnchorBlockSizeY * numAnchorBlockY + (BIY == GDY - 1) * (SPLINE_DIM >= 2);
    auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (BIZ == GDZ - 1) * (SPLINE_DIM >= 3);
    auto TOTAL = x_size * y_size * z_size;

    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
        auto x   = (_tix % x_size);
        auto y   = (_tix / x_size) % y_size;
        auto z   = (_tix / x_size) / y_size;
        auto gx  = (x + BIX * AnchorBlockSizeX * numAnchorBlockX);
        auto gy  = (y + BIY * AnchorBlockSizeY * numAnchorBlockY);
        auto gz  = (z + BIZ * AnchorBlockSizeZ * numAnchorBlockZ);
        auto gid = gx + gy * buf_leap.y + gz * buf_leap.z;

        if (gx < buf_size.x and gy < buf_size.y and gz < buf_size.z) dram_buf[gid] = s_buf[z][y][x];
    }
    __syncthreads();
}


template <typename T1, typename T2, int LEVEL = 4,
int SPLINE_DIM = 2, int AnchorBlockSizeX = 8,
int AnchorBlockSizeY = 8, int AnchorBlockSizeZ = 8,
int numAnchorBlockX = 4,  // Number of Anchor blocks along X
int numAnchorBlockY = 1,  // Number of Anchor blocks along Y
int numAnchorBlockZ = 1,  // Number of Anchor blocks along Z
int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void
shmem2global_data_with_compaction(volatile T1 s_buf[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
[AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
[AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)], T2* dram_buf, DIM3 buf_size, STRIDE3 buf_leap, int radius,volatile size_t grid_leaps[LEVEL + 1][2], volatile size_t prefix_nums[LEVEL + 1], T1* dram_compactval = nullptr, uint32_t* dram_compactidx = nullptr, uint32_t* dram_compactnum = nullptr)
{
    auto x_size = AnchorBlockSizeX * numAnchorBlockX + (BIX == GDX - 1) * (SPLINE_DIM >= 1);
    auto y_size = AnchorBlockSizeY * numAnchorBlockY + (BIY == GDY - 1) * (SPLINE_DIM >= 2);
    auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (BIZ == GDZ - 1) * (SPLINE_DIM >= 3);
    auto TOTAL = x_size * y_size * z_size;

    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
        auto x   = (_tix % x_size);
        auto y   = (_tix / x_size) % y_size;
        auto z   = (_tix / x_size) / y_size;
        auto gx  = (x + BIX * AnchorBlockSizeX * numAnchorBlockX);
        auto gy  = (y + BIY * AnchorBlockSizeY * numAnchorBlockY);
        auto gz  = (z + BIZ * AnchorBlockSizeZ * numAnchorBlockZ);
        //auto gid = gx + gy * buf_leap.y + gz * buf_leap.z;

        auto candidate = s_buf[z][y][x];
        bool quantizable = (candidate >= 0) and (candidate < 2*radius);

        if (gx < buf_size.x and gy < buf_size.y and gz < buf_size.z) {
            if (not quantizable) {
                auto data_gid = gx + gy * buf_leap.y + gz * buf_leap.z;
                auto cur_idx = atomicAdd(dram_compactnum, 1);
                dram_compactidx[cur_idx] = data_gid;
                dram_compactval[cur_idx] = candidate;
            }
            int level = 0;
            //todo: pre-compute the leaps and their halves
            while(gx % 2 == 0 and gy % 2 == 0 and gz % 2 == 0 and level < LEVEL){
                gx = gx >> 1;
                gy = gy >> 1;
                gz = gz >> 1;
                level++;
            }
            auto gid = gx + gy * grid_leaps[level][0] + gz * grid_leaps[level][1];

            if(level < LEVEL){//non-anchor
                gid += prefix_nums[level]-((gz + 1) >> 1) * grid_leaps[level + 1][1] - (gz % 2 == 0) * ((gy + 1) >> 1) * grid_leaps[level + 1][0] - (gz % 2 == 0 && gy % 2 == 0) * ((gx + 1) >> 1);
            }

            // TODO this is for algorithmic demo by reading from shmem
            // For performance purpose, it can be inlined in quantization
            dram_buf[gid] = quantizable * static_cast<T2>(candidate);
        }
    }
}



template <
    typename T1,
    typename T2,
    typename FP,
    int SPLINE_DIM, int AnchorBlockSizeX,
    int AnchorBlockSizeY, int AnchorBlockSizeZ,
    int numAnchorBlockX,
    int numAnchorBlockY,
    int numAnchorBlockZ,
    typename LAMBDAX,
    typename LAMBDAY,
    typename LAMBDAZ,
    bool BLUE,
    bool YELLOW,
    bool HOLLOW,
    bool COARSEN,
    int  LINEAR_BLOCK_SIZE,
    bool BORDER_INCLUSIVE,
    bool WORKFLOW>
__forceinline__ __device__ void interpolate_stage(
    volatile T1 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                       [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                       [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    volatile T2 s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                    [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    DIM3    data_size,
    LAMBDAX     xmap,
    LAMBDAY     ymap,
    LAMBDAZ     zmap,
    int         unit,
    FP          eb_r,
    FP          ebx2,
    int         radius,
    bool interpolator,
    int  BLOCK_DIMX,
    int  BLOCK_DIMY,
    int  BLOCK_DIMZ)
{
    // static_assert(BLOCK_DIMX * BLOCK_DIMY * (COARSEN ? 1 : BLOCK_DIMZ) <= BLOCK_DIM_SIZE, "block oversized");
    static_assert((BLUE or YELLOW or HOLLOW) == true, "must be one hot");
    static_assert((BLUE and YELLOW) == false, "must be only one hot (1)");
    static_assert((BLUE and YELLOW) == false, "must be only one hot (2)");
    static_assert((YELLOW and HOLLOW) == false, "must be only one hot (3)");


    auto run = [&](auto x, auto y, auto z) {
        if (xyz_predicate<SPLINE_DIM,
            AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
            numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ,
            BORDER_INCLUSIVE>(x, y, z,data_size)) {
            auto global_x = BIX * AnchorBlockSizeX * numAnchorBlockX + x;
            auto global_y = BIY * AnchorBlockSizeY * numAnchorBlockY + y;
            auto global_z = BIZ * AnchorBlockSizeZ * numAnchorBlockZ + z;  
            
            T1 pred = 0;
            auto input_x = x;
            auto input_BI = BIX;
            auto input_GD = GDX;
            auto input_gx = global_x;
            auto input_gs = data_size.x;
            auto right_bound = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
            auto x_size = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
            auto y_size = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
            // auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
            int p1 = -1, p2 = 9, p3 = 9, p4 = -1, p5 = 16;
            if(interpolator==0){
                p1 = -3, p2 = 23, p3 = 23, p4 = -3, p5 = 40;
            }
            if CONSTEXPR (BLUE){
                input_x = z;
                input_BI = BIZ;
                input_GD = GDZ;
                input_gx = global_z;
                input_gs = data_size.z;
                right_bound = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
            }
            if CONSTEXPR (YELLOW){
                input_x = y;
                input_BI = BIY;
                input_GD = GDY;
                input_gx = global_y;
                input_gs = data_size.y;
                right_bound = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
            }
            
            int id_[4], s_id[4];
            id_[0] =  input_x - 3 * unit;
            id_[0] =  id_[0] >= 0 ? id_[0] : 0;
        
            id_[1] = input_x - unit;
            id_[1] = id_[1] >= 0 ? id_[1] : 0;
        
            id_[2] = input_x + unit;
            id_[2] = id_[2] < right_bound ? id_[2] : 0;
            
            id_[3] = input_x + 3 * unit;
            id_[3] = id_[3] < right_bound ? id_[3] : 0;
            
            s_id[0] = x_size * y_size * z + x_size * y + id_[0];
            s_id[1] = x_size * y_size * z + x_size * y + id_[1];
            s_id[2] = x_size * y_size * z + x_size * y + id_[2];
            s_id[3] = x_size * y_size * z + x_size * y + id_[3];
            if CONSTEXPR (BLUE){
            s_id[0] = x_size * y_size * id_[0] + x_size * y + x;
            s_id[1] = x_size * y_size * id_[1] + x_size * y + x;
            s_id[2] = x_size * y_size * id_[2] + x_size * y + x;
            s_id[3] = x_size * y_size * id_[3] + x_size * y + x;
            }
            if CONSTEXPR (YELLOW){
                s_id[0] = x_size * y_size * z + x_size * id_[0] + x;
                s_id[1] = x_size * y_size * z + x_size * id_[1] + x;
                s_id[2] = x_size * y_size * z + x_size * id_[2] + x;
                s_id[3] = x_size * y_size * z + x_size * id_[3] + x;
            }

        
            bool case1 = (input_BI != input_GD - 1);
            bool case2 = (input_x >= 3 * unit);
            bool case3 = (input_x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX);
            bool case4 = (input_gx + 3 * unit < input_gs);
            bool case5 = (input_gx + unit < input_gs);
            
            // 预加载 shared memory 数据到寄存器
            T1 tmp0 = *((T1*)s_data + s_id[0]); 
            T1 tmp1 = *((T1*)s_data + s_id[1]); 
            T1 tmp2 = *((T1*)s_data + s_id[2]); 
            T1 tmp3 = *((T1*)s_data + s_id[3]); 

            // 初始预测值
            pred = tmp1;

            // 计算不同 case 对应的 pred
            if ((case1 && !case2 && !case3) || (!case1 && !case2 && !(case3 && case4) && case5)) {
                pred = (tmp1 + tmp2) / 2;
            }
            else if ((case1 && !case2 && case3) || (!case1 && !case2 && case3 && case4)) {
                pred = (3 * tmp1 + 6 * tmp2 - tmp3) / 8;
            }
            else if ((case1 && case2 && !case3) || (!case1 && case2 && !(case3 && case4) && case5)) {
                pred = (-tmp0 + 6 * tmp1 + 3 * tmp2) / 8;
            }
            else if ((case1 && case2 && case3) || (!case1 && case2 && case3 && case4)) {
                pred = (p1 * tmp0 + p2 * tmp1 + p3 * tmp2 + p4 * tmp3) / p5;
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
    auto TOTAL = BLOCK_DIMX * BLOCK_DIMY * BLOCK_DIMZ;
    if CONSTEXPR (COARSEN) {
        
        //if( BLOCK_DIMX *BLOCK_DIMY<= LINEAR_BLOCK_SIZE){
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
        if(TIX < TOTAL){
            auto itix = (TIX % BLOCK_DIMX);
            auto itiy = (TIX / BLOCK_DIMX) % BLOCK_DIMY;
            auto itiz = (TIX / BLOCK_DIMX) / BLOCK_DIMY;
            auto x    = xmap(itix, unit);
            auto y    = ymap(itiy, unit);
            auto z    = zmap(itiz, unit);


            run(x, y, z);
        }
    }
    __syncthreads();
}

template <
    typename T1,
    typename T2,
    typename FP,
    int SPLINE_DIM, int AnchorBlockSizeX,
    int AnchorBlockSizeY, int AnchorBlockSizeZ,
    int numAnchorBlockX,
    int numAnchorBlockY,
    int numAnchorBlockZ,
    typename LAMBDA,
    bool LINE,
    bool FACE,
    bool CUBE,
    int  LINEAR_BLOCK_SIZE,
    bool COARSEN,
    bool BORDER_INCLUSIVE,
    bool WORKFLOW,
    typename INTERP>
__forceinline__ __device__ void interpolate_stage_md(
    volatile T1 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
    [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
volatile T2 s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
 [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
 [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    DIM3    data_size,
    LAMBDA xyzmap,
    int         unit,
    FP          eb_r,
    FP          ebx2,
    int         radius,
    INTERP cubic_interpolator,
    int NUM_ELE)
{
    // static_assert(COARSEN or (NUM_ELE <= BLOCK_DIM_SIZE), "block oversized");
    static_assert((LINE or FACE or CUBE) == true, "must be one hot");
    static_assert((LINE and FACE) == false, "must be only one hot (1)");
    static_assert((LINE and CUBE) == false, "must be only one hot (2)");
    static_assert((FACE and CUBE) == false, "must be only one hot (3)");

    auto run = [&](auto x, auto y, auto z) {

        if (xyz_predicate<SPLINE_DIM,
            AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
            numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, BORDER_INCLUSIVE>(x, y, z,data_size)) {
            T1 pred = 0;
            auto global_x = BIX * AnchorBlockSizeX * numAnchorBlockX + x;
            auto global_y = BIY * AnchorBlockSizeY * numAnchorBlockY + y;
            auto global_z = BIZ * AnchorBlockSizeZ * numAnchorBlockZ + z;  

          
           int id_z[4], id_y[4], id_x[4];
           id_z[0] = (z - 3 * unit >= 0) ? z - 3 * unit : 0;
           id_z[1] = (z - unit >= 0) ? z - unit : 0;
           id_z[2] = (z + unit <= AnchorBlockSizeZ * numAnchorBlockZ) ? z + unit : 0;
           id_z[3] = (z + 3 * unit <= AnchorBlockSizeZ * numAnchorBlockZ) ? z + 3 * unit : 0;
           
           id_y[0] = (y - 3 * unit >= 0) ? y - 3 * unit : 0;
           id_y[1] = (y - unit >= 0) ? y - unit : 0;
           id_y[2] = (y + unit <= AnchorBlockSizeY * numAnchorBlockY) ? y + unit : 0;
           id_y[3] = (y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY) ? y + 3 * unit : 0;
           
           id_x[0] = (x - 3 * unit >= 0) ? x - 3 * unit : 0;
           id_x[1] = (x - unit >= 0) ? x - unit : 0;
           id_x[2] = (x + unit <= AnchorBlockSizeX * numAnchorBlockX) ? x + unit : 0;
           id_x[3] = (x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX) ? x + 3 * unit : 0;
           
            if CONSTEXPR (LINE) {
                
                bool I_Y = (y % (2*unit) )> 0; 
                bool I_Z = (z % (2*unit) )> 0; 

                pred = 0;
                auto input_x = x;
                auto input_BI = BIX;
                auto input_GD = GDX;
                auto input_gx = global_x;
                auto input_gs = data_size.x;

                auto right_bound = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
                auto x_size = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
                auto y_size = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
                // auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
                
                if (I_Z){
                    input_x = z;
                    input_BI = BIZ;
                    input_GD = GDZ;
                    input_gx = global_z;
                    input_gs = data_size.z;
                    right_bound = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
                }
                else if (I_Y){
                    input_x = y;
                    input_BI = BIY;
                    input_GD = GDY;
                    input_gx = global_y;
                    input_gs = data_size.y;
                    right_bound = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
                }
                
                int id_[4], s_id[4];
                id_[0] =  input_x - 3 * unit;
                id_[0] =  id_[0] >= 0 ? id_[0] : 0;
            
                id_[1] = input_x - unit;
                id_[1] = id_[1] >= 0 ? id_[1] : 0;
            
                id_[2] = input_x + unit;
                id_[2] = id_[2] < right_bound ? id_[2] : 0;
                
                id_[3] = input_x + 3 * unit;
                id_[3] = id_[3] < right_bound ? id_[3] : 0;
                
                s_id[0] = x_size * y_size * z + x_size * y + id_[0];
                s_id[1] = x_size * y_size * z + x_size * y + id_[1];
                s_id[2] = x_size * y_size * z + x_size * y + id_[2];
                s_id[3] = x_size * y_size * z + x_size * y + id_[3];
                if (I_Z){
                s_id[0] = x_size * y_size * id_[0] + x_size * y + x;
                s_id[1] = x_size * y_size * id_[1] + x_size * y + x;
                s_id[2] = x_size * y_size * id_[2] + x_size * y + x;
                s_id[3] = x_size * y_size * id_[3] + x_size * y + x;
                }
                else if (I_Y){
                    s_id[0] = x_size * y_size * z + x_size * id_[0] + x;
                    s_id[1] = x_size * y_size * z + x_size * id_[1] + x;
                    s_id[2] = x_size * y_size * z + x_size * id_[2] + x;
                    s_id[3] = x_size * y_size * z + x_size * id_[3] + x;
                }

            
                bool case1 = (input_BI != input_GD - 1);
                bool case2 = (input_x >= 3 * unit);
                bool case3 = (input_x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX);
                bool case4 = (input_gx + 3 * unit < input_gs);
                bool case5 = (input_gx + unit < input_gs);
                
                
                // 预加载 shared memory 数据到寄存器
                T1 tmp0 = *((T1*)s_data + s_id[0]); 
                T1 tmp1 = *((T1*)s_data + s_id[1]); 
                T1 tmp2 = *((T1*)s_data + s_id[2]); 
                T1 tmp3 = *((T1*)s_data + s_id[3]); 
    
                // 初始预测值
                pred = tmp1;
    
                // 计算不同 case 对应的 pred
                if ( (case1 && case2 && case3) || (!case1 && case2 && case3 && case4)) {
                    pred = cubic_interpolator(tmp0, tmp1, tmp2, tmp3);
                    
                }
                else if ((case1 && case2 && !case3) || ( !case1 && case2 && !(case3 && case4) && case5)) {
                    pred = (-tmp0 + 6 * tmp1 + 3 * tmp2) / 8;
                }
                else if ((case1 && !case2 && case3) || (!case1 && !case2 && case3 && case4 )){
                    pred = (3 * tmp1 + 6 * tmp2 - tmp3) / 8;   
                }
                else if ((case1 && !case2 && !case3) || (!case1 && !case2 && !(case3 && case4) && case5)) {
                    pred = (tmp1 + tmp2) / 2;
                }

            }
            auto get_interp_order = [&](auto x, auto BI, auto GD, auto gx, auto gs){
                int b = (x >= 3 * unit) ? 3 : 1;
                int f = ((x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX) && ((BI != GD - 1) || (gx + 3 * unit < gs))) ? 3 :
                (((BI != GD - 1) || (gx + unit < gs)) ? 1 : 0);

                return (b == 3) ? ((f == 3) ? 4 : ((f == 1) ? 3 : 0)) 
                                : ((f == 3) ? 2 : ((f == 1) ? 1 : 0));
            };
            if CONSTEXPR (FACE) {  //
               // if(BIX == 5 and BIY == 22 and BIZ == 6 and unit==1 and x==29 and y==7 and z==0){
               //     printf("%.2e %.2e %.2e %.2e\n",s_data[z ][y- 3*unit][x],s_data[z ][y- unit][x],s_data[z ][y+ unit][x]);
              //  }

                bool I_YZ = (x % (2*unit) ) == 0;
                bool I_XZ = (y % (2*unit ) )== 0;

                //if(BIX == 10 and BIY == 12 and BIZ == 0 and x==13 and y==6 and z==9)
               //     printf("face %d %d\n", I_YZ,I_XZ);
                int x_1,BI_1,GD_1,gx_1,gs_1;
                int x_2,BI_2,GD_2,gx_2,gs_2;
                int s_id_1[4], s_id_2[4];
                auto x_size = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
                auto y_size = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
                // auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
                if (I_YZ){
                   
                 x_1 = z,BI_1 = BIZ, GD_1 = GDZ, gx_1 = global_z,gs_1 = data_size.z;
                 x_2 = y,BI_2 = BIY, GD_2 = GDY, gx_2 = global_y, gs_2 = data_size.y;
                 s_id_1[0] = x_size * y_size * id_z[0] + x_size * y + x;
                 s_id_1[1] = x_size * y_size * id_z[1] + x_size * y + x;
                 s_id_1[2] = x_size * y_size * id_z[2] + x_size * y + x;
                 s_id_1[3] = x_size * y_size * id_z[3] + x_size * y + x;
                 s_id_2[0] = x_size * y_size * z + x_size * id_y[0] + x;
                 s_id_2[1] = x_size * y_size * z + x_size * id_y[1] + x;
                 s_id_2[2] = x_size * y_size * z + x_size * id_y[2] + x;
                 s_id_2[3] = x_size * y_size * z + x_size * id_y[3] + x;
                 pred = s_data[id_z[1]][id_y[1]][x];

                }
                else if (I_XZ){
                    x_1 = z,BI_1 = BIZ, GD_1 = GDZ, gx_1 = global_z,gs_1 = data_size.z;
                    x_2 = x,BI_2 = BIX, GD_2 = GDX, gx_2 = global_x, gs_2 = data_size.x;
                    s_id_1[0] = x_size * y_size * id_z[0] + x_size * y + x;
                    s_id_1[1] = x_size * y_size * id_z[1] + x_size * y + x;
                    s_id_1[2] = x_size * y_size * id_z[2] + x_size * y + x;
                    s_id_1[3] = x_size * y_size * id_z[3] + x_size * y + x;
                    
                    s_id_2[0] = x_size * y_size * z + x_size * y + id_x[0];
                    s_id_2[1] = x_size * y_size * z + x_size * y + id_x[1];
                    s_id_2[2] = x_size * y_size * z + x_size * y + id_x[2];
                    s_id_2[3] = x_size * y_size * z + x_size * y + id_x[3];
                    pred = s_data[id_z[1]][y][id_x[1]];
                    
                }
                else{
                    x_1 = y,BI_1 = BIY, GD_1 = GDY, gx_1 = global_y, gs_1 = data_size.y;
                    x_2 = x,BI_2 = BIX, GD_2 = GDX, gx_2 = global_x, gs_2 = data_size.x;
                    s_id_1[0] = x_size * y_size * z + x_size * id_y[0] + x;
                    s_id_1[1] = x_size * y_size * z + x_size * id_y[1] + x;
                    s_id_1[2] = x_size * y_size * z + x_size * id_y[2] + x;
                    s_id_1[3] = x_size * y_size * z + x_size * id_y[3] + x;
                    s_id_2[0] = x_size * y_size * z + x_size * y + id_x[0];
                    s_id_2[1] = x_size * y_size * z + x_size * y + id_x[1];
                    s_id_2[2] = x_size * y_size * z + x_size * y + id_x[2];
                    s_id_2[3] = x_size * y_size * z + x_size * y + id_x[3];
                    pred = s_data[z][id_y[1]][id_x[1]];
                }

                    auto interp_1 = get_interp_order(x_1,BI_1,GD_1,gx_1,gs_1);
                    auto interp_2 = get_interp_order(x_2,BI_2,GD_2,gx_2,gs_2);

                    int case_num = interp_1 + interp_2 * 5;


                    if (interp_1 == 4 && interp_2 == 4) {
                        pred = (cubic_interpolator(*((T1*)s_data + s_id_1[0]), 
                        *((T1*)s_data + s_id_1[1]), 
                        *((T1*)s_data + s_id_1[2]), 
                        *((T1*)s_data + s_id_1[3])) +
                         cubic_interpolator(*((T1*)s_data + s_id_2[0]), 
                        *((T1*)s_data + s_id_2[1]), 
                        *((T1*)s_data + s_id_2[2]), 
                        *((T1*)s_data + s_id_2[3]))) / 2;
                    } else if (interp_1 != 4 && interp_2 == 4) {
                        pred = cubic_interpolator(*((T1*)s_data + s_id_2[0]), 
                        *((T1*)s_data + s_id_2[1]), 
                        *((T1*)s_data + s_id_2[2]), 
                        *((T1*)s_data + s_id_2[3]));
                    } else if (interp_1 == 4 && interp_2 != 4) {
                        pred = cubic_interpolator(*((T1*)s_data + s_id_1[0]), 
                        *((T1*)s_data + s_id_1[1]), 
                        *((T1*)s_data + s_id_1[2]), 
                        *((T1*)s_data + s_id_1[3]));
                    } else if (interp_1 == 3 && interp_2 == 3) {
                        pred = (-(*((T1*)s_data + s_id_2[0]))+6*(*((T1*)s_data + s_id_2[1])) + 3*(*((T1*)s_data + s_id_2[2]))) / 8;
                        pred += (-(*((T1*)s_data + s_id_1[0]))+6*(*((T1*)s_data + s_id_1[1])) + 3*(*((T1*)s_data + s_id_1[2]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 3 && interp_2 == 2) {
                        pred = (3*(*((T1*)s_data + s_id_2[1]))+6*(*((T1*)s_data + s_id_2[2])) - (*((T1*)s_data + s_id_2[3]))) / 8;
                        pred += (-(*((T1*)s_data + s_id_1[0]))+6*(*((T1*)s_data + s_id_1[1])) + 3*(*((T1*)s_data + s_id_1[2]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 3 && interp_2 < 2) {
                        pred = (-(*((T1*)s_data + s_id_1[0]))+6*(*((T1*)s_data + s_id_1[1])) + 3*(*((T1*)s_data + s_id_1[2]))) / 8;
                    } else if (interp_1 == 2 && interp_2 == 3) {
                        pred = (3*(*((T1*)s_data + s_id_1[1]))+6*(*((T1*)s_data + s_id_1[2])) - (*((T1*)s_data + s_id_1[3]))) / 8;
                        pred += (-(*((T1*)s_data + s_id_2[0]))+6*(*((T1*)s_data + s_id_2[1])) + 3*(*((T1*)s_data + s_id_2[2]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 2 && interp_2 == 2) {
                        pred = (3*(*((T1*)s_data + s_id_1[1]))+6*(*((T1*)s_data + s_id_1[2])) - (*((T1*)s_data + s_id_1[3]))) / 8;
                        pred += (3*(*((T1*)s_data + s_id_2[1]))+6*(*((T1*)s_data + s_id_2[2])) - (*((T1*)s_data + s_id_2[3]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 2 && interp_2 < 2) {
                        pred = (3*(*((T1*)s_data + s_id_1[1]))+6*(*((T1*)s_data + s_id_1[2])) - (*((T1*)s_data + s_id_1[3]))) / 8;
                    } else if (interp_1 <= 1 && interp_2 == 3) {
                        pred = (-(*((T1*)s_data + s_id_2[0]))+6*(*((T1*)s_data + s_id_2[1])) + 3*(*((T1*)s_data + s_id_2[2]))) / 8;
                    } else if (interp_1 <= 1 && interp_2 == 2) {
                        pred = (3*(*((T1*)s_data + s_id_2[1]))+6*(*((T1*)s_data + s_id_2[2])) - (*((T1*)s_data + s_id_2[3]))) / 8;
                    } else if (interp_1 == 1 && interp_2 == 1) {
                        pred = ((*((T1*)s_data + s_id_2[1]))+(*((T1*)s_data + s_id_2[2]))) / 2;
                        pred += ((*((T1*)s_data + s_id_1[1]))+(*((T1*)s_data + s_id_1[2]))) / 2;
                        pred /= 2;
                    } else if (interp_1 == 1 && interp_2 < 1) {
                        
                        pred = ((*((T1*)s_data + s_id_1[1]))+(*((T1*)s_data + s_id_1[2]))) / 2;
                    } else if (interp_1 == 0 && interp_2 == 1) {
                        pred = ((*((T1*)s_data + s_id_2[1]))+(*((T1*)s_data + s_id_2[2]))) / 2;
                    }
                    else{
                        pred = (*((T1*)s_data + s_id_1[1])) + (*((T1*)s_data + s_id_2[1])) - pred;
                    }
                    
            }

            if CONSTEXPR (CUBE) {  //
                T1 tmp_z[4], tmp_y[4], tmp_x[4];
                auto interp_z = get_interp_order(z,BIZ,GDZ,global_z,data_size.z);
                auto interp_y = get_interp_order(y,BIY,GDY,global_y,data_size.y);
                auto interp_x = get_interp_order(x,BIX,GDX,global_x,data_size.x);
                
                #pragma unroll
                for(int id_itr = 0; id_itr < 4; ++id_itr){
                 tmp_x[id_itr] = s_data[z][y][id_x[id_itr]]; 
                }
                if(interp_z == 4){
                    #pragma unroll
                    for(int id_itr = 0; id_itr < 4; ++id_itr){
                        tmp_z[id_itr] = s_data[id_z[id_itr]][y][x];
                       }
                }
                if(interp_y == 4){
                    #pragma unroll
                    for(int id_itr = 0; id_itr < 4; ++id_itr){
                     tmp_y[id_itr] = s_data[z][id_y[id_itr]][x]; 
                    }
                }


                T1 pred_z[5], pred_y[5], pred_x[5];
                pred_x[0] = tmp_x[1];
                pred_x[1] = cubic_interpolator(tmp_x[0],tmp_x[1],tmp_x[2],tmp_x[3]);
                pred_x[2] = (-tmp_x[0]+6*tmp_x[1] + 3*tmp_x[2]) / 8;
                pred_x[3] = (3*tmp_x[1] + 6*tmp_x[2]-tmp_x[3]) / 8;
                pred_x[4] = (tmp_x[1] + tmp_x[2]) / 2;
                
                pred_y[1] = cubic_interpolator(tmp_y[0],tmp_y[1],tmp_y[2],tmp_y[3]);

                
                pred_z[1] = cubic_interpolator(tmp_z[0],tmp_z[1],tmp_z[2],tmp_z[3]);
                
                pred = pred_x[0];
                pred = (interp_z == 4 && interp_y == 4 && interp_x == 4) ? (pred_x[1] +  pred_y[1] + pred_z[1]) / 3 : pred;
                
                pred = (interp_z == 4 && interp_y == 4 && interp_x != 4) ? (pred_z[1] + pred_y[1]) / 2 : pred;
                pred = (interp_z == 4 && interp_y != 4 && interp_x == 4) ? (pred_z[1] + pred_x[1]) / 2 : pred;
                pred = (interp_z != 4 && interp_y == 4 && interp_x == 4) ? (pred_y[1] + pred_x[1]) / 2 : pred;
                
                pred = (interp_z == 4 && interp_y != 4 && interp_x != 4) ? pred_z[1]: pred;
                pred = (interp_z != 4 && interp_y == 4 && interp_x != 4) ? pred_y[1]: pred;
                pred = (interp_z != 4 && interp_y != 4 && interp_x == 4) ? pred_x[1]: pred;


                pred = (interp_z != 4 && interp_y != 4 && interp_x == 3) ? pred_x[2]: pred;
                pred = (interp_z != 4 && interp_y != 4 && interp_x == 2) ? pred_x[3]: pred;
                pred = (interp_z != 4 && interp_y != 4 && interp_x == 1) ? pred_x[4]: pred;
                // pred = (interp_z != 4 && interp_y != 4 && interp_x == 0) ? pred_x[0]: pred;
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
        auto TOTAL = NUM_ELE;
        for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
            auto [x,y,z]    = xyzmap(_tix, unit);
            run(x, y, z);
        }
        
    }
    else {
        if(TIX<NUM_ELE){
            auto [x,y,z]    = xyzmap(TIX, unit);
            run(x, y, z);
        }
    }
    __syncthreads();
}

}  // namespace

template <typename T, int SPLINE_DIM = 3,
int PROFILE_BLOCK_SIZE_X = 4,
int PROFILE_BLOCK_SIZE_Y = 4,
int PROFILE_BLOCK_SIZE_Z = 4,
int PROFILE_NUM_BLOCK_X = 4,
int PROFILE_NUM_BLOCK_Y = 4,
int PROFILE_NUM_BLOCK_Z = 4,
int  LINEAR_BLOCK_SIZE>
__device__ void cusz::device_api::auto_tuning(volatile T s_data[PROFILE_BLOCK_SIZE_Z * PROFILE_NUM_BLOCK_Z]       
    [PROFILE_BLOCK_SIZE_Y * PROFILE_NUM_BLOCK_Y][PROFILE_BLOCK_SIZE_X * PROFILE_NUM_BLOCK_X],  volatile T local_errs[2], DIM3  data_size,  T * errs){
 
    if(TIX < 2)
        local_errs[TIX] = 0;
    __syncthreads(); 

    auto local_idx = TIX % 2;
    auto temp = TIX / 2;
    

    auto block_idx_x =  temp % PROFILE_NUM_BLOCK_X;
    auto block_idx_y = ( temp / PROFILE_NUM_BLOCK_X) % PROFILE_NUM_BLOCK_Y;
    auto block_idx_z = ( ( temp  / PROFILE_NUM_BLOCK_X) / PROFILE_NUM_BLOCK_Y) % PROFILE_NUM_BLOCK_Z;
    auto dir = ( ( temp  / PROFILE_NUM_BLOCK_X) / PROFILE_NUM_BLOCK_Y) / PROFILE_NUM_BLOCK_Z;



    bool predicate = dir < 2;

    if(predicate){
        auto x = PROFILE_BLOCK_SIZE_X * block_idx_x + 1 + local_idx;
        auto y = PROFILE_BLOCK_SIZE_Y * block_idx_y + 1 + local_idx;
        auto z = PROFILE_BLOCK_SIZE_Z * block_idx_z + 1 + local_idx;
        T pred = 0;
        switch(dir){
            case 0:
                pred = (s_data[z - 1][y][x] + s_data[z + 1][y][x]) / 2;
                break;
            case 1:
                pred = (s_data[z][y][x - 1] + s_data[z][y][x + 1]) / 2;
                break;
            default:
            break;
        }
        
        T abs_error = fabs(pred - s_data[z][y][x]);
        atomicAdd(const_cast<T*>(local_errs) + dir, abs_error);
    } 
    __syncthreads(); 
    if(TIX < 2)
        errs[TIX]=local_errs[TIX];
    __syncthreads(); 
       
}

template <typename T,
int SPLINE_DIM = 3,
int PROFILE_NUM_BLOCK_X = 4,
int PROFILE_NUM_BLOCK_Y = 4,
int PROFILE_NUM_BLOCK_Z = 4, 
int  LINEAR_BLOCK_SIZE>
__device__ void cusz::device_api::auto_tuning_2(volatile T s_data[PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z], volatile T s_nx[PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z][4], volatile T s_ny[PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z][4], volatile T s_nz[PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z][4],  volatile T local_errs[6], DIM3  data_size,  T * errs){
    
    if CONSTEXPR (SPLINE_DIM == 3){
        if(TIX<6)
            local_errs[TIX]=0;
        __syncthreads(); 

        auto point_idx = TIX % (PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z);
        auto c = TIX / (PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z);


        bool predicate = c < 6;
        if(predicate){

            T pred=0;

            //auto unit = 1;
            switch(c){
                    case 0:
                        pred = (-s_nz[point_idx][0] + 9 * s_nz[point_idx][1] + 9 * s_nz[point_idx][2] - s_nz[point_idx][3]) / 16;
                        break;

                    case 1:
                        pred = (-3 * s_nz[point_idx][0] + 23 * s_nz[point_idx][1] + 23* s_nz[point_idx][2] - 3 * s_nz[point_idx][3]) / 40;
                        break;
                    case 2:
                        pred = (-s_ny[point_idx][0] + 9 * s_ny[point_idx][1] + 9 * s_ny[point_idx][2] - s_ny[point_idx][3]) / 16;
                        break;
                    case 3:
                        pred = (-3 * s_ny[point_idx][0] + 23 * s_ny[point_idx][1] + 23 * s_ny[point_idx][2] - 3 * s_ny[point_idx][3]) / 40;
                        break;

                    case 4:
                        pred = (-s_nx[point_idx][0] + 9 * s_nx[point_idx][1] + 9 * s_nx[point_idx][2] - s_nx[point_idx][3]) / 16;
                        break;
                    case 5:
                        pred = (-3 * s_nx[point_idx][0] + 23 * s_nx[point_idx][1] + 23 * s_nx[point_idx][2] - 3 * s_nx[point_idx][3]) / 40;
                        break;
                    default:
                    break;
                }
            T abs_error=fabs(pred-s_data[point_idx]);
            atomicAdd(const_cast<T*>(local_errs) + c, abs_error);
        } 
        __syncthreads(); 
        if(TIX<6)
            errs[TIX]=local_errs[TIX];
        __syncthreads(); 
    }
    
    if CONSTEXPR (SPLINE_DIM == 3){
        if(TIX<4)
            local_errs[TIX]=0;
        __syncthreads(); 
        auto point_idx = TIX % (PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z);
        auto c = TIX / (PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z);
        bool predicate = c < 4;
        if(predicate){
            T pred=0;
            switch(c){
                case 0:
                    pred = (-s_ny[point_idx][0] + 9 * s_ny[point_idx][1] + 9 * s_ny[point_idx][2] - s_ny[point_idx][3]) / 16;
                    break;
                case 1:
                    pred = (-3 * s_ny[point_idx][0] + 23 * s_ny[point_idx][1] + 23 * s_ny[point_idx][2] - 3 * s_ny[point_idx][3]) / 40;
                    break;
                case 2:
                    pred = (-s_nx[point_idx][0] + 9 * s_nx[point_idx][1] + 9 * s_nx[point_idx][2] - s_nx[point_idx][3]) / 16;
                    break;
                case 3:
                    pred = (-3 * s_nx[point_idx][0] + 23 * s_nx[point_idx][1] + 23 * s_nx[point_idx][2] - 3 * s_nx[point_idx][3]) / 40;
                    break;
                default:
                break;
            }
            T abs_error=fabs(pred-s_data[point_idx]);
            atomicAdd(const_cast<T*>(local_errs) + c, abs_error);
        } 
        __syncthreads(); 
        if(TIX<4)
            errs[TIX]=local_errs[TIX];
        __syncthreads(); 
    }       
}

template<int SPLINE_DIM, int BLOCKSIZE>
__device__ std::tuple<int, int, int> xyzmap_line(int _tix, const int UNIT);
template<int SPLINE_DIM, int BLOCKSIZE>
__device__ std::tuple<int, int, int> xyzmap_face(int _tix, const int UNIT);
template<int SPLINE_DIM, int BLOCKSIZE>
__device__ std::tuple<int, int, int> xyzmap_cube(int _tix, const int UNIT);


template<int SPLINE_DIM, int BLOCKSIZE>
__device__ std::tuple<int, int, int> xyzmap_line(int _tix, const int UNIT) {
    if constexpr (SPLINE_DIM == 3) {
        auto N = BLOCKSIZE / (UNIT * 2);
        auto L = N * (N+1) * (N+1); 
        auto Q = (N+1) * (N+1); 
        auto group = _tix / L ;
        auto m = _tix % L ;
        auto i = m / Q;
        auto j = (m % Q) / (N+1);
        auto k = (m % Q) % (N+1);
        if(group == 0)
            return std::make_tuple(2 * UNIT * i + UNIT, 2 * UNIT * j, 2 * UNIT * k);
        else if (group == 1)
            return std::make_tuple(2 * UNIT * k, 2 * UNIT * i + UNIT, 2 * UNIT * j);
        else
            return std::make_tuple(2 * UNIT * j, 2 * UNIT * k, 2 * UNIT * i + UNIT);
    }
    if constexpr (SPLINE_DIM == 2) {
        auto N = BLOCKSIZE / (UNIT * 2);
        auto L = N * (N+1); 
        auto Q = (N+1); 
        auto group = _tix / L ;
        auto m = _tix % L ;
        auto i = m / Q;
        auto j = (m % Q);
        if(group == 0)
            return std::make_tuple(2 * UNIT * i + UNIT, 2 * UNIT * j, 0);
        else
            return std::make_tuple(2 * UNIT * j, 2 * UNIT * i + UNIT, 0);
    }
}

template<int SPLINE_DIM, int BLOCKSIZE>
__device__ std::tuple<int, int, int> xyzmap_face(int _tix, const int UNIT) {
    if constexpr (SPLINE_DIM == 3) {
        auto N = BLOCKSIZE / (UNIT * 2);
        auto L = N * N * (N+1);
        auto Q = N * N; 
        auto group = _tix / L ;
        auto m = _tix % L ;
        auto i = m / Q;
        auto j = (m % Q) / N;
        auto k = (m % Q) % N;
        if(group == 0)
            return std::make_tuple(2 * UNIT * i, 2 * UNIT * j + UNIT, 2 * UNIT * k + UNIT);
        else if (group == 1)
            return std::make_tuple(2 * UNIT * k + UNIT, 2 * UNIT * i, 2 * UNIT * j + UNIT);
        else
            return std::make_tuple(2 * UNIT * j + UNIT, 2 * UNIT * k + UNIT, 2 * UNIT * i);
    }
    if constexpr (SPLINE_DIM == 2) {
        auto N = BLOCKSIZE / (UNIT * 2);
        auto L = N * N;
        auto Q = N * N; 
        // auto group = _tix / L ;
        auto m = _tix % L ;
        
        auto i = (m % Q) / N;
        auto j = (m % Q) % N;
        return std::make_tuple(2 * UNIT * i + UNIT, 2 * UNIT * j + UNIT, 0);
    }
}


template<int SPLINE_DIM, int BLOCKSIZE>
__device__ std::tuple<int, int, int> xyzmap_cube(int _tix, const int UNIT) {
    if constexpr (SPLINE_DIM == 3) {
        auto N = BLOCKSIZE / (UNIT * 2);
        auto Q = N * N; 
        auto i = _tix / Q;
        auto j = (_tix % Q) / N;
        auto k = (_tix % Q) % N;
        return std::make_tuple(2 * UNIT * i + UNIT, 2 * UNIT * j + UNIT, 2 * UNIT * k + UNIT);
    }
}



template <typename T1, typename T2, typename FP, int LEVEL, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ, int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ, int LINEAR_BLOCK_SIZE, bool WORKFLOW, bool PROBE_PRED_ERROR>
__device__ void cusz::device_api::spline_layout_interpolate(
    volatile T1 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                       [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                       [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    volatile T2 s_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                    [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    DIM3    data_size,
    FP          eb_r,
    FP          ebx2,
    int         radius,
    INTERPOLATION_PARAMS intp_param
    )
{
    auto xblue = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2); };
    auto yblue = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
    auto zblue = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2 + 1); };

    auto xblue_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix ); };
    auto yblue_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy ); };
    auto zblue_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2 + 1); };

    auto xyellow = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2); };
    auto yyellow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2+1); };
    auto zyellow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };

    auto xyellow_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix ); };
    auto yyellow_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2+1); };
    auto zyellow_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2); };


    auto xhollow = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2 +1); };
    auto yhollow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy); };
    auto zhollow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };

    auto xhollow_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2 +1); };
    auto yhollow_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
    auto zhollow_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz *2); };

    auto nan_cubic_interp = [] __device__ (T1 a, T1 b, T1 c, T1 d) -> T1{
        return (-a+9*b+9*c-d) / 16;
    };

    auto nat_cubic_interp = [] __device__ (T1 a, T1 b, T1 c, T1 d) -> T1{
        return (-3*a+23*b+23*c-3*d) / 40;
    };

    constexpr auto COARSEN          = true;
    // constexpr auto NO_COARSEN       = false;
    constexpr auto BORDER_INCLUSIVE = true;
    constexpr auto BORDER_EXCLUSIVE = false;

    
    FP cur_ebx2=ebx2,cur_eb_r=eb_r;


    auto calc_eb = [&](auto unit) {
        cur_ebx2=ebx2,cur_eb_r=eb_r;
        int temp=1;
        while(temp<unit){
            temp*=2;
            cur_eb_r *= intp_param.alpha;
            cur_ebx2 /= intp_param.alpha;

        }
        if(cur_ebx2 < ebx2 / intp_param.beta){
            cur_ebx2 = ebx2 / intp_param.beta;
            cur_eb_r = eb_r * intp_param.beta;

        }
    };
    
    int max_unit = ((AnchorBlockSizeX >= AnchorBlockSizeY) ? AnchorBlockSizeX : AnchorBlockSizeY);
    max_unit = ((max_unit >= AnchorBlockSizeZ) ? max_unit : AnchorBlockSizeZ);
    max_unit /= 2;
    int unit_x = AnchorBlockSizeX, unit_y = AnchorBlockSizeY, unit_z = AnchorBlockSizeZ;
    int level_id = LEVEL;
    level_id -= 1;
    #pragma unroll
    for(int unit = max_unit; unit >= 1; unit /= 2, level_id--){
        calc_eb(unit);
        unit_x = (SPLINE_DIM >= 1) ? unit * 2 : 1;
        unit_y = (SPLINE_DIM >= 2) ? unit * 2 : 1;
        unit_z = (SPLINE_DIM >= 3) ? unit * 2 : 1;
        if(level_id != 0){
            if(intp_param.use_md[level_id]){
                int N_x = AnchorBlockSizeX / (unit * 2);
                int N_y = AnchorBlockSizeY / (unit * 2);
                int N_z = AnchorBlockSizeZ / (unit * 2);
                int N_line = N_x * (N_y + 1) * (N_z + 1) + (N_x + 1) * N_y * (N_z + 1) + (N_x + 1) * (N_y + 1) * N_z;
                int N_face = N_x * N_y * (N_z + 1) + N_x * (N_y + 1) * N_z + (N_x + 1) * N_y * N_z; 
                int N_cube = N_x * N_y * N_z;
                if(intp_param.use_natural[level_id]==0){
                    if constexpr (SPLINE_DIM >= 1)
                    interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>), true, false, false, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp, N_line);
                    if constexpr (SPLINE_DIM >= 2)
                    interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>), false, true, false, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp, N_face);
                    if constexpr (SPLINE_DIM >= 3)
                    interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>), false, false, true, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp, N_cube);
                }
                else{
                    if constexpr (SPLINE_DIM >= 1)
                    interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>), true, false, false, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp, N_line);
                    if constexpr (SPLINE_DIM >= 2)
                    interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>), false, true, false, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp, N_face);
                    if constexpr (SPLINE_DIM >= 3)
                    interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>), false, false, true, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp, N_cube);
                }
            }
            else{
                if(intp_param.reverse[level_id]){
                    if constexpr (SPLINE_DIM >= 1) {
                    interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse), false, false, true, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], numAnchorBlockX * AnchorBlockSizeX / unit_x, numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                    unit_x /= 2;
                    }
                    if constexpr (SPLINE_DIM >= 2) {
                    interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse), false, true, false, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y,numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                    unit_y /= 2;
                    }
                    if constexpr (SPLINE_DIM >= 3) {
                    interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse), true, false, false, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xblue_reverse, yblue_reverse, zblue_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z);
                    unit_z /= 2;
                    }
                }
                else{
                    if constexpr (SPLINE_DIM >= 3) {
                    interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xblue), decltype(yblue), decltype(zblue), true, false, false, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xblue, yblue, zblue, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z);
                    unit_z /= 2;
                    }
                    if constexpr (SPLINE_DIM >= 2) {
                    interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyellow), decltype(yyellow), decltype(zyellow), false, true, false, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xyellow, yyellow, zyellow, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y, numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                    unit_y /= 2;
                    }
                    if constexpr (SPLINE_DIM >= 1) {
                    interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xhollow), decltype(yhollow), decltype(zhollow), false, false, true, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xhollow, yhollow, zhollow, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], numAnchorBlockX * AnchorBlockSizeX / unit_x, numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                    unit_x /= 2;
                    }
                }
        }

        }
        else{
            if(intp_param.use_md[level_id]){
                int N_x = AnchorBlockSizeX / (unit * 2);
                int N_y = AnchorBlockSizeY / (unit * 2);
                int N_z = AnchorBlockSizeZ / (unit * 2);
                int N_line = N_x * (N_y + 1) * (N_z + 1) + (N_x + 1) * N_y * (N_z + 1) + (N_x + 1) * (N_y + 1) * N_z;
                int N_face = N_x * N_y * (N_z + 1) + N_x * (N_y + 1) * N_z + (N_x + 1) * N_y * N_z; 
                int N_cube = N_x * N_y * N_z;
                if(intp_param.use_natural[level_id]==0){
                    if constexpr (SPLINE_DIM >= 1)
                    interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>), true, false, false, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp, N_line);
                    if constexpr (SPLINE_DIM >= 2)
                    interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>), false, true, false, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp, N_face);
                    if constexpr (SPLINE_DIM >= 3)
                    interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>), false, false, true, LINEAR_BLOCK_SIZE, COARSEN, BORDER_EXCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nan_cubic_interp, N_cube);
                }
                else{
                    if constexpr (SPLINE_DIM >= 1)
                    interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>), true, false, false, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp, N_line);
                    if constexpr (SPLINE_DIM >= 2)
                    interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>), false, true, false, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp, N_face);
                    if constexpr (SPLINE_DIM >= 3)
                    interpolate_stage_md<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>), false, false, true, LINEAR_BLOCK_SIZE, COARSEN, BORDER_EXCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, radius, nat_cubic_interp, N_cube);
                }
            }
            else{
                if(intp_param.reverse[level_id]){
                    if constexpr (SPLINE_DIM >= 1) {
                    interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse), false, false, true, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], numAnchorBlockX * AnchorBlockSizeX / unit_x, numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                    unit_x /= 2;
                    }
                    if constexpr (SPLINE_DIM >= 2) {
                    interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse), false, true, false, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y,numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                    unit_y /= 2;
                    }
                    if constexpr (SPLINE_DIM >= 3) {
                    interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse), true, false, false, COARSEN, LINEAR_BLOCK_SIZE, BORDER_EXCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xblue_reverse, yblue_reverse, zblue_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z);
                    unit_z /= 2;
                    }
                }
                else{
                    if constexpr (SPLINE_DIM >= 3) {
                    interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xblue), decltype(yblue), decltype(zblue), true, false, false, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xblue, yblue, zblue, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z);
                    unit_z /= 2;
                    }
                    if constexpr (SPLINE_DIM >= 2) {
                    interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyellow), decltype(yyellow), decltype(zyellow), false, true, false, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xyellow, yyellow, zyellow, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y, numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                    unit_y /= 2;
                    }
                    if constexpr (SPLINE_DIM >= 1) {
                    interpolate_stage<T1, T2, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xhollow), decltype(yhollow), decltype(zhollow), false, false, true, COARSEN, LINEAR_BLOCK_SIZE, BORDER_EXCLUSIVE, WORKFLOW>(s_data, s_ectrl,data_size, xhollow, yhollow, zhollow, unit, cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id], numAnchorBlockX * AnchorBlockSizeX / unit_x, numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                    unit_x /= 2;
                    }
                }
        }

        }
    }

}

/********************************************************************************
 * host API/kernel
 ********************************************************************************/
template <typename TITER, int SPLINE_DIM = 3,
int PROFILE_BLOCK_SIZE_X = 4,
int PROFILE_BLOCK_SIZE_Y = 4,
int PROFILE_BLOCK_SIZE_Z = 4,
int PROFILE_NUM_BLOCK_X = 4,
int PROFILE_NUM_BLOCK_Y = 4,
int PROFILE_NUM_BLOCK_Z = 4, 
int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__global__ void cusz::c_spline_profiling_data(
    TITER   data,
    DIM3    data_size,
    STRIDE3 data_leap,
    TITER errors)
{
    // compile time variables
    using T = typename std::remove_pointer<TITER>::type;
 

    {
        
        __shared__    T shmem_data[PROFILE_BLOCK_SIZE_Z * PROFILE_NUM_BLOCK_Z][PROFILE_BLOCK_SIZE_Y * PROFILE_NUM_BLOCK_Y][PROFILE_BLOCK_SIZE_X * PROFILE_NUM_BLOCK_X];
        __shared__    T shmem_local_errs[2];
           
        // __shared__ struct {
        //     T data[PROFILE_BLOCK_SIZE_Z * PROFILE_NUM_BLOCK_Z][PROFILE_BLOCK_SIZE_Y * PROFILE_NUM_BLOCK_Y][PROFILE_BLOCK_SIZE_X * PROFILE_NUM_BLOCK_X];
        //     T local_errs[2];
        //    // T global_errs[6];
        // } shmem;
        c_reset_scratch_profiling_data<T, SPLINE_DIM, PROFILE_BLOCK_SIZE_X, PROFILE_BLOCK_SIZE_Y, PROFILE_BLOCK_SIZE_Z, PROFILE_NUM_BLOCK_X, PROFILE_NUM_BLOCK_Y, PROFILE_NUM_BLOCK_Z, LINEAR_BLOCK_SIZE>(shmem_data, 0.0);
        
        global2shmem_profiling_data<T, T, PROFILE_BLOCK_SIZE_X, PROFILE_BLOCK_SIZE_Y, PROFILE_BLOCK_SIZE_Z, PROFILE_NUM_BLOCK_X, PROFILE_NUM_BLOCK_Y, PROFILE_NUM_BLOCK_Z, LINEAR_BLOCK_SIZE>(data, data_size, data_leap, shmem_data);

        cusz::device_api::auto_tuning<T, SPLINE_DIM, PROFILE_BLOCK_SIZE_X, PROFILE_BLOCK_SIZE_Y, PROFILE_BLOCK_SIZE_Z, PROFILE_NUM_BLOCK_X, PROFILE_NUM_BLOCK_Y, PROFILE_NUM_BLOCK_Z, LINEAR_BLOCK_SIZE>(shmem_data, shmem_local_errs, data_size, errors);

    }
}

template <typename TITER, int SPLINE_DIM = 3, int PROFILE_NUM_BLOCK_X = 4, int PROFILE_NUM_BLOCK_Y = 4, int PROFILE_NUM_BLOCK_Z = 4, int  LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__global__ void cusz::c_spline_profiling_data_2(
    TITER   data,
    DIM3    data_size,
    STRIDE3 data_leap,
    TITER errors)
{
    // compile time variables
    using T = typename std::remove_pointer<TITER>::type;
 

    {
        
        __shared__ T shmem_data[PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z];
        __shared__ T shmem_neighbor_x [PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z][4];
        __shared__ T shmem_neighbor_y [PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z][4];
        __shared__ T shmem_neighbor_z [PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z][4];
        __shared__ T shmem_local_errs[6];
           // T global_errs[6];
        
        // __shared__ struct {
        //     T data[PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z];
        //     T neighbor_x [PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z][4];
        //     T neighbor_y [PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z][4];
        //     T neighbor_z [PROFILE_NUM_BLOCK_X * PROFILE_NUM_BLOCK_Y * PROFILE_NUM_BLOCK_Z][4];
        //     T local_errs[6];
        //    // T global_errs[6];
        // } shmem;


        c_reset_scratch_profiling_data_2<T, SPLINE_DIM, PROFILE_NUM_BLOCK_X, PROFILE_NUM_BLOCK_Y, PROFILE_NUM_BLOCK_Z, LINEAR_BLOCK_SIZE>(shmem_data, shmem_neighbor_x, shmem_neighbor_y, shmem_neighbor_z,0.0);
        global2shmem_profiling_data_2<T, T, SPLINE_DIM, PROFILE_NUM_BLOCK_X, PROFILE_NUM_BLOCK_Y, PROFILE_NUM_BLOCK_Z, LINEAR_BLOCK_SIZE>(data, data_size, data_leap,shmem_data, shmem_neighbor_x, shmem_neighbor_y, shmem_neighbor_z);

        if (TIX < 6 and BIX==0 and BIY==0 and BIZ==0) errors[TIX] = 0.0;//risky

        cusz::device_api::auto_tuning_2<T, SPLINE_DIM, PROFILE_NUM_BLOCK_X, PROFILE_NUM_BLOCK_Y, PROFILE_NUM_BLOCK_Z, LINEAR_BLOCK_SIZE>(
            shmem_data, shmem_neighbor_x, shmem_neighbor_y, shmem_neighbor_z, shmem_local_errs, data_size, errors);

        
    }
}


template <int LEVEL> __forceinline__ __device__ void pre_compute(DIM3 data_size, volatile size_t grid_leaps[LEVEL + 1][2], volatile size_t prefix_nums[LEVEL + 1]){
    if(TIX==0){
        auto d_size = data_size;
        
        int level = 0;
        while(level <= LEVEL){
            //grid_leaps[level][0] = 1;
            grid_leaps[level][0] = d_size.x;
            grid_leaps[level][1] = d_size.x * d_size.y;
            if(level < LEVEL){
                d_size.x = (d_size.x + 1) >> 1;
                d_size.y = (d_size.y + 1) >> 1;
                d_size.z = (d_size.z + 1) >> 1;
                prefix_nums[level] = d_size.x * d_size.y * d_size.z;
            }
            level++;
        }   
        prefix_nums[LEVEL] = 0;
    }
    __syncthreads(); 
}

template <typename TITER, typename EITER, typename FP,  int LEVEL, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ, int numAnchorBlockX, 
int numAnchorBlockY, int numAnchorBlockZ, int LINEAR_BLOCK_SIZE, typename CompactVal, typename CompactIdx, typename CompactNum>
__global__ void cusz::c_spline_infprecis_data(
    TITER   data,
    DIM3    data_size,
    STRIDE3 data_leap,
    EITER   ectrl,
    DIM3    ectrl_size,
    STRIDE3 ectrl_leap,
    TITER   anchor,
    STRIDE3 anchor_leap,
    CompactVal compact_val,
    CompactIdx compact_idx,
    CompactNum compact_num,
    FP      eb_r,
    FP      ebx2,
    int     radius,
    INTERPOLATION_PARAMS intp_param//,
    )
{
    // compile time variables
    using T = typename std::remove_pointer<TITER>::type;
    using E = typename std::remove_pointer<EITER>::type;

    {
        // __shared__ struct {
            __shared__ T shmem_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
             [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
             [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
             __shared__ T shmem_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
                    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
                    [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
                __shared__ size_t shmem_grid_leaps[LEVEL + 1][2];
                __shared__ size_t shmem_prefix_nums[LEVEL + 1];
        // } shmem;

   
        pre_compute<LEVEL>(ectrl_size, shmem_grid_leaps, shmem_prefix_nums);

        c_reset_scratch_data<T, T, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE>(shmem_data, shmem_ectrl, radius);

        global2shmem_data<T, T, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE>(data, data_size, data_leap, shmem_data);

        c_gather_anchor<T, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ>(data, data_size, data_leap, anchor, anchor_leap);
        cusz::device_api::spline_layout_interpolate<T, T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_COMPR, false>(
            shmem_data, shmem_ectrl, data_size, eb_r, ebx2, radius, intp_param);

        shmem2global_data_with_compaction<T, E, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY,  numAnchorBlockZ, LINEAR_BLOCK_SIZE>(shmem_ectrl, ectrl, ectrl_size, ectrl_leap, radius, shmem_grid_leaps,shmem_prefix_nums, compact_val, compact_idx, compact_num);
    }
}

template <
    typename EITER,
    typename TITER,
    typename FP, 
    int LEVEL,
    int SPLINE_DIM, int AnchorBlockSizeX,
    int AnchorBlockSizeY, int AnchorBlockSizeZ,
    int numAnchorBlockX,  // Number of Anchor blocks along X
    int numAnchorBlockY,  // Number of Anchor blocks along Y
    int numAnchorBlockZ,  // Number of Anchor blocks along Z
    int LINEAR_BLOCK_SIZE>
__global__ void cusz::x_spline_infprecis_data(
    EITER   ectrl,        // input 1
    DIM3    ectrl_size,   //
    STRIDE3 ectrl_leap,   //
    TITER   anchor,       // input 2
    DIM3    anchor_size,  //
    STRIDE3 anchor_leap,  //
    TITER   data,         // output
    DIM3    data_size,    //
    STRIDE3 data_leap,    //
    TITER   outlier_tmp,
    FP      eb_r,
    FP      ebx2,
    int     radius,
    INTERPOLATION_PARAMS intp_param)
{
    // compile time variables
    using E = typename std::remove_pointer<EITER>::type;
    using T = typename std::remove_pointer<TITER>::type;

    // __shared__ struct {
    //     T data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
    //     [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
    //     [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
    //     T ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
    //     [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
    //     [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
    //     STRIDE3 grid_leaps[LEVEL + 1];
    //     size_t prefix_nums[LEVEL + 1];
    // } shmem;

    __shared__ T shmem_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
    [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
    __shared__ T shmem_ectrl[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
           [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
           [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
    __shared__ size_t shmem_grid_leaps[LEVEL + 1][2];
    __shared__ size_t shmem_prefix_nums[LEVEL + 1];

    pre_compute<LEVEL>(ectrl_size, shmem_grid_leaps, shmem_prefix_nums);

    x_reset_scratch_data<T, T, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE>(shmem_data, shmem_ectrl, anchor, anchor_size, anchor_leap);
    global2shmem_fuse<T, E, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE>(ectrl, ectrl_size, ectrl_leap, outlier_tmp, shmem_ectrl, shmem_grid_leaps, shmem_prefix_nums);

    cusz::device_api::spline_layout_interpolate<T, T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_DECOMPR, false>(
        shmem_data, shmem_ectrl, data_size, eb_r, ebx2, radius, intp_param);
    shmem2global_data<T, T, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE>(shmem_data, data, data_size, data_leap);
}


template <typename TITER>
__global__ void cusz::reset_errors(TITER errors)
{
    if(TIX<36)
        errors[TIX]=0;
}

template <typename T, int SPLINE_DIM>
__forceinline__ __device__ void pre_compute_att(DIM3 sam_starts, DIM3 sam_bgs, DIM3 sam_strides, DIM3 &global_starts, INTERPOLATION_PARAMS &intp_param, uint8_t &level, uint8_t &unit, volatile T err[6], bool workflow);

// template <typename T>
// __forceinline__ __device__ void pre_compute_att<T, 3>(DIM3 sam_starts, DIM3 sam_bgs, DIM3 sam_strides, DIM3 &global_starts, INTERPOLATION_PARAMS &intp_param, uint8_t &level, uint8_t &unit, volatile T err[6], bool workflow){
template <typename T, int SPLINE_DIM, int LEVEL>
__forceinline__ __device__ void pre_compute_att(DIM3 sam_starts, DIM3 sam_bgs, DIM3 sam_strides, DIM3 &global_starts, INTERPOLATION_PARAMS &intp_param, uint8_t &level, uint8_t &unit, volatile T err[9], bool workflow){

    if(TIX < 9) err[TIX] = 0.0;

    auto grid_idx_x = BIX % sam_bgs.x;
    auto grid_idx_y = (BIX / sam_bgs.x) % sam_bgs.y;
    auto grid_idx_z = (BIX / sam_bgs.x) / sam_bgs.y;
    global_starts.x = sam_starts.x + grid_idx_x * sam_strides.x;
    global_starts.y = sam_starts.y + grid_idx_y * sam_strides.y;
    global_starts.z = sam_starts.z + grid_idx_z * sam_strides.z;
    
    if CONSTEXPR (SPLINE_DIM == 3){
        if(workflow == SPLINE3_PRED_ATT){
            bool use_natural = false, use_md = false, reverse = false;
            if (BIY == 0){
                level = 2;
            }
            else if (BIY < 3){
                level = 1;
                use_natural = (BIY == 2);
            }
            else{
                level = 0;
                use_natural = BIY > 5;
                use_md = (BIY == 5 or BIY == 8);
                reverse = BIY % 3;
            }       
            intp_param.use_natural[level] = use_natural;
            intp_param.use_md[level] = use_md;
            intp_param.reverse[level] = reverse;
        }
        else{
            level = 0;
            if(BIY == 0){
                intp_param.alpha = 1.0;
                intp_param.beta = 2.0;
            }
            else if (BIY == 1){
                intp_param.alpha = 1.25;
                intp_param.beta = 2.0;
            }
            else{
                intp_param.alpha = 1.5 + 0.25 * ((BIY - 2) / 3);
                intp_param.beta = 2.0 + ((BIY - 2) % 3);
            }
        }
        unit = 1 << level;
    }

    if CONSTEXPR (SPLINE_DIM == 2){
    
        if(workflow == SPLINE3_PRED_ATT){
            // bool use_natural = false, use_md = false, reverse = false;
            // level = LEVEL - (BIY / 6) - 1;
            // use_natural = (BIY % 6) >= 3;
            // use_md = (BIY % 3) == 2;
            // reverse = BIY % 3;
            // intp_param.use_natural[level] = use_natural;
            // intp_param.use_md[level] = use_md;
            // intp_param.reverse[level] = reverse;
            bool use_natural = false, use_md = false, reverse = false;
            if (BIY == 0){
                level = 3;
            }
            else if (BIY < 3){
                level = 2;
                use_natural = (BIY == 2);
            }
            else if (BIY < 5){
                level = 1;
                use_natural = (BIY == 2);
            }
            else{
                level = 0;
                use_natural = BIY > 7;
                use_md = (BIY == 7 or BIY == 10);
                reverse = (BIY + 1) % 3;
            }       
            intp_param.use_natural[level] = use_natural;
            intp_param.use_md[level] = use_md;
            intp_param.reverse[level] = reverse;
        }
        else{
            level = 0;
            if(BIY == 0){
                intp_param.alpha = 1.0;
                intp_param.beta = 2.0;
            }
            else if (BIY == 1){
                intp_param.alpha = 1.25;
                intp_param.beta = 2.0;
            }
            else{
                intp_param.alpha = 1.5 + 0.25 * ((BIY - 2) / 3);
                intp_param.beta = 2.0 + ((BIY - 2) % 3);
            }
        }
        unit = 1 << level;
    }
    
    __syncthreads();
}


template <typename T1, typename T2, int SPLINE_DIM = 2, int AnchorBlockSizeX = 8, int AnchorBlockSizeY = 8, int AnchorBlockSizeZ = 8, int numAnchorBlockX = 4, int numAnchorBlockY = 1,  int numAnchorBlockZ = 1, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void global2shmem_data_att(T1* data, DIM3 data_size, STRIDE3 data_leap,   volatile T2 s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
[AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
[AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)], DIM3 global_starts, uint8_t unit)
{
    constexpr auto TOTAL = (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)) *
    (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)) * 
    (AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3));

    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
        auto x = (_tix % (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)));
        auto y = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) %
                 (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
        auto z = (_tix / (AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1))) /
                 (AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2));
        auto gx  = (x + global_starts.x);
        auto gy  = (y + global_starts.y);
        auto gz  = (z + global_starts.z);
        auto gid = gx + gy * data_leap.y + gz * data_leap.z;

        if (gx < data_size.x and gy < data_size.y and gz < data_size.z  ) s_data[z][y][x] = data[gid];
    }
    __syncthreads();
}


template <
    typename T,
    typename FP,
    int SPLINE_DIM, int AnchorBlockSizeX,
    int AnchorBlockSizeY, int AnchorBlockSizeZ,
    int numAnchorBlockX,
    int numAnchorBlockY,
    int numAnchorBlockZ,
    typename LAMBDAX,
    typename LAMBDAY,
    typename LAMBDAZ,
    bool BLUE,
    bool YELLOW,
    bool HOLLOW,
    bool COARSEN,
    int  LINEAR_BLOCK_SIZE,
    bool BORDER_INCLUSIVE,
    bool WORKFLOW
    >
__forceinline__ __device__ void interpolate_stage_att(
    volatile T s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)][AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)][AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    DIM3    data_size,
     DIM3    global_starts,
    LAMBDAX     xmap,
    LAMBDAY     ymap,
    LAMBDAZ     zmap,
    int         unit,
    FP eb_r,
    FP ebx2,
    bool interpolator,
    volatile T* error,
    int  BLOCK_DIMX,
    int  BLOCK_DIMY,
    int  BLOCK_DIMZ)
{
    
    // static_assert(BLOCK_DIMX * BLOCK_DIMY * (COARSEN ? 1 : BLOCK_DIMZ) <= BLOCK_DIM_SIZE, "block oversized");
    static_assert((BLUE or YELLOW or HOLLOW) == true, "must be one hot");
    static_assert((BLUE and YELLOW) == false, "must be only one hot (1)");
    static_assert((BLUE and YELLOW) == false, "must be only one hot (2)");
    static_assert((YELLOW and HOLLOW) == false, "must be only one hot (3)");
    //DIM3 global_starts (global_starts_v.x,global_starts_v.y, global_starts_v.z);
    auto run = [&](auto x, auto y, auto z) {
        if (xyz_predicate_att<SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, BORDER_INCLUSIVE>(x, y, z,data_size, global_starts)) {
            T pred = 0;

            auto global_x=global_starts.x+x, global_y=global_starts.y+y, global_z=global_starts.z+z;
            auto input_x = x;
            // auto input_BI = BIX;
            // auto input_GD = GDX;
            auto input_gx = global_x;
            auto input_gs = data_size.x;
            auto right_bound = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
            auto x_size = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
            auto y_size = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
            // auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
            int global_start_ = global_starts.x;
            int p1 = -1, p2 = 9, p3 = 9, p4 = -1, p5 = 16;
            if(interpolator==0){
               p1 = -3, p2 = 23, p3 = 23, p4 = -3, p5 = 40;
           }
           if CONSTEXPR (BLUE){
               input_x = z;
            //    input_BI = BIZ;
            //    input_GD = GDZ;
               input_gx = global_z;
               input_gs = data_size.z;
               global_start_ = global_starts.z;
               right_bound = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
           }
           if CONSTEXPR (YELLOW){
               input_x = y;
            //    input_BI = BIY;
            //    input_GD = GDY;
               input_gx = global_y;
               input_gs = data_size.y;
               global_start_ = global_starts.y;
               right_bound = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
           }
           
           int id_[4], s_id[4];
           id_[0] =  input_x - 3 * unit;
           id_[0] =  id_[0] >= 0 ? id_[0] : 0;
           
           id_[1] = input_x - unit;
           id_[1] = id_[1] >= 0 ? id_[1] : 0;
           
           id_[2] = input_x + unit;
           id_[2] = id_[2] < right_bound ? id_[2] : 0;
           
           id_[3] = input_x + 3 * unit;
           id_[3] = id_[3] < right_bound ? id_[3] : 0;
           
           s_id[0] = x_size * y_size * z + x_size * y + id_[0];
           s_id[1] = x_size * y_size * z + x_size * y + id_[1];
           s_id[2] = x_size * y_size * z + x_size * y + id_[2];
           s_id[3] = x_size * y_size * z + x_size * y + id_[3];
           if CONSTEXPR (BLUE){
            s_id[0] = x_size * y_size * id_[0] + x_size * y + x;
            s_id[1] = x_size * y_size * id_[1] + x_size * y + x;
            s_id[2] = x_size * y_size * id_[2] + x_size * y + x;
            s_id[3] = x_size * y_size * id_[3] + x_size * y + x;
           }
           if CONSTEXPR (YELLOW){
            s_id[0] = x_size * y_size * z + x_size * id_[0] + x;
            s_id[1] = x_size * y_size * z + x_size * id_[1] + x;
            s_id[2] = x_size * y_size * z + x_size * id_[2] + x;
            s_id[3] = x_size * y_size * z + x_size * id_[3] + x;
           }
           
           
           bool case1 = (global_start_ + AnchorBlockSizeX * numAnchorBlockX < input_gs);
           bool case2 = (input_x >= 3 * unit);
           bool case3 = (input_x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX);
           bool case4 = (input_gx + 3 * unit < input_gs);
           bool case5 = (input_gx + unit < input_gs);
           
           
           // 预加载 shared memory 数据到寄存器
           T tmp0 = *((T*)s_data + s_id[0]); 
           T tmp1 = *((T*)s_data + s_id[1]); 
           T tmp2 = *((T*)s_data + s_id[2]); 
           T tmp3 = *((T*)s_data + s_id[3]); 
           
           // 初始预测值
           pred = tmp1;
           
           // 计算不同 case 对应的 pred
           if ((case1 && !case2 && !case3) || (!case1 && !case2 && !(case3 && case4) && case5)) {
               pred = (tmp1 + tmp2) / 2;
           }
           else if ((case1 && !case2 && case3) || (!case1 && !case2 && case3 && case4)) {
               pred = (3 * tmp1 + 6 * tmp2 - tmp3) / 8;
           }
           else if ((case1 && case2 && !case3) || (!case1 && case2 && !(case3 && case4) && case5)) {
               pred = (-tmp0 + 6 * tmp1 + 3 * tmp2) / 8;
           }
           else if ((case1 && case2 && case3) || (!case1 && case2 && case3 && case4)) {
               pred = (p1 * tmp0 + p2 * tmp1 + p3 * tmp2 + p4 * tmp3) / p5;
           }

             if CONSTEXPR (WORKFLOW == SPLINE3_AB_ATT) {
                
                auto          err = s_data[z][y][x] - pred;
                decltype(err) code;
                // TODO unsafe, did not deal with the out-of-cap case
                {
                    code = fabs(err) * eb_r + 1;
                    code = err < 0 ? -code : code;
                    code = int(code / 2) ;
                }
                
                s_data[z][y][x]  = pred + code * ebx2;
                atomicAdd(const_cast<T*>(error),code!=0);
                

            }
            else{
                // if(TIX == 0 and BIX == 0) printf("BIY=%d s_data[%d][%d][%d]=%f, pred=%f\n", BIY, z, y, x, s_data[z][y][x], pred);
                atomicAdd(const_cast<T*>(error),fabs(s_data[z][y][x]-pred));
            }
        }
    };
    // -------------------------------------------------------------------------------- //
    auto TOTAL = BLOCK_DIMX * BLOCK_DIMY * BLOCK_DIMZ;
    // if(TIX == 0 and BIX == 0) printf("interpolate_stage_att BIY=%d, BLOCK_DIMX=%d, BLOCK_DIMY=%d, BLOCK_DIMZ=%d, TOTAL=%d\n", BIY, BLOCK_DIMX, BLOCK_DIMY, BLOCK_DIMZ, TOTAL);
    if CONSTEXPR (COARSEN) {
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
        if (TIX < TOTAL){
            auto itix = (TIX % BLOCK_DIMX);
            auto itiy = (TIX / BLOCK_DIMX) % BLOCK_DIMY;
            auto itiz = (TIX / BLOCK_DIMX) / BLOCK_DIMY;
            auto x    = xmap(itix, unit);
            auto y    = ymap(itiy, unit);
            auto z    = zmap(itiz, unit);
            run(x, y, z);
        }
    }
    __syncthreads();
}

template <
    typename T,
    typename FP,
    int SPLINE_DIM, int AnchorBlockSizeX,
    int AnchorBlockSizeY, int AnchorBlockSizeZ,
    int numAnchorBlockX,
    int numAnchorBlockY,
    int numAnchorBlockZ,
    typename LAMBDA,
    bool LINE,
    bool FACE,
    bool CUBE,
    int  LINEAR_BLOCK_SIZE,
    bool COARSEN,
    bool BORDER_INCLUSIVE,
    bool WORKFLOW,
    typename INTERP>
__forceinline__ __device__ void interpolate_stage_md_att(
    volatile T s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)][AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)][AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    DIM3    data_size,
    DIM3    global_starts,
    LAMBDA xyzmap,
    int         unit,
    FP eb_r,
    FP ebx2,
    INTERP cubic_interpolator,
    volatile T* error,
    int NUM_ELE)
{
    // static_assert(COARSEN or (NUM_ELE <= BLOCK_DIM_SIZE), "block oversized");
    static_assert((LINE or FACE or CUBE) == true, "must be one hot");
    static_assert((LINE and FACE) == false, "must be only one hot (1)");
    static_assert((LINE and CUBE) == false, "must be only one hot (2)");
    static_assert((FACE and CUBE) == false, "must be only one hot (3)");
    //DIM3 global_starts (global_starts_v.x,global_starts_v.y, global_starts_v.z);
    auto run = [&](auto x, auto y, auto z) {
        if (xyz_predicate_att<SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, BORDER_INCLUSIVE>(x, y, z, data_size, global_starts)) {
            T pred = 0;

            auto global_x=global_starts.x+x, global_y=global_starts.y+y, global_z=global_starts.z+z;
        //    T tmp_z[4], tmp_y[4], tmp_x[4];
           int id_z[4], id_y[4], id_x[4];
           id_z[0] = (z - 3 * unit >= 0) ? z - 3 * unit : 0;
           id_z[1] = (z - unit >= 0) ? z - unit : 0;
           id_z[2] = (z + unit <=  AnchorBlockSizeZ * numAnchorBlockZ) ? z + unit : 0;
           id_z[3] = (z + 3 * unit <=  AnchorBlockSizeZ * numAnchorBlockZ) ? z + 3 * unit : 0;
           
           id_y[0] = (y - 3 * unit >= 0) ? y - 3 * unit : 0;
           id_y[1] = (y - unit >= 0) ? y - unit : 0;
           id_y[2] = (y + unit <= AnchorBlockSizeY * numAnchorBlockY) ? y + unit : 0;
           id_y[3] = (y + 3 * unit <= AnchorBlockSizeY * numAnchorBlockY) ? y + 3 * unit : 0;
           
           id_x[0] = (x - 3 * unit >= 0) ? x - 3 * unit : 0;
           id_x[1] = (x - unit >= 0) ? x - unit : 0;
           id_x[2] = (x + unit <= AnchorBlockSizeX * numAnchorBlockX) ? x + unit : 0;
           id_x[3] = (x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX) ? x + 3 * unit : 0;
           
            if CONSTEXPR (LINE) {
                bool I_Y = (y % (2*unit) )> 0; 
                bool I_Z = (z % (2*unit) )> 0; 

                pred = 0;
                auto input_x = x;
                //auto input_BI = BIX;
                //auto input_GD = GDX;
                auto input_gx = global_x;
                auto input_gs = data_size.x;
                auto right_bound = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
                auto x_size = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
                auto y_size = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
                // auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
                int global_start_ = global_starts.x;
                if (I_Z){
                    input_x = z;
                    //input_BI = BIZ;
                    //input_GD = GDZ;
                    input_gx = global_z;
                    input_gs = data_size.z;
                    global_start_ = global_starts.z;
                    right_bound = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
                }
                else if (I_Y){
                    input_x = y;
                    //input_BI = BIY;
                    //input_GD = GDY;
                    input_gx = global_y;
                    input_gs = data_size.y;
                    global_start_ = global_starts.y;
                    right_bound = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
                }
                
                int id_[4], s_id[4];
                id_[0] =  input_x - 3 * unit;
                id_[0] =  id_[0] >= 0 ? id_[0] : 0;
            
                id_[1] = input_x - unit;
                id_[1] = id_[1] >= 0 ? id_[1] : 0;
            
                id_[2] = input_x + unit;
                id_[2] = id_[2] < right_bound ? id_[2] : 0;
                
                id_[3] = input_x + 3 * unit;
                id_[3] = id_[3] < right_bound ? id_[3] : 0;
                
                s_id[0] = x_size * y_size * z + x_size * y + id_[0];
                s_id[1] = x_size * y_size * z + x_size * y + id_[1];
                s_id[2] = x_size * y_size * z + x_size * y + id_[2];
                s_id[3] = x_size * y_size * z + x_size * y + id_[3];
                if (I_Z){
                    s_id[0] = x_size * y_size * id_[0] + x_size * y + x;
                    s_id[1] = x_size * y_size * id_[1] + x_size * y + x;
                    s_id[2] = x_size * y_size * id_[2] + x_size * y + x;
                    s_id[3] = x_size * y_size * id_[3] + x_size * y + x;
                }
                else if (I_Y){
                    s_id[0] = x_size * y_size * z + x_size * id_[0] + x;
                    s_id[1] = x_size * y_size * z + x_size * id_[1] + x;
                    s_id[2] = x_size * y_size * z + x_size * id_[2] + x;
                    s_id[3] = x_size * y_size * z + x_size * id_[3] + x;
                }

            
                bool case1 = (global_start_ + AnchorBlockSizeX * numAnchorBlockX < input_gs);
                bool case2 = (input_x >= 3 * unit);
                bool case3 = (input_x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX);
                bool case4 = (input_gx + 3 * unit < input_gs);
                bool case5 = (input_gx + unit < input_gs);
                
                
                // 预加载 shared memory 数据到寄存器
                T tmp0 = *((T*)s_data + s_id[0]); 
                T tmp1 = *((T*)s_data + s_id[1]); 
                T tmp2 = *((T*)s_data + s_id[2]); 
                T tmp3 = *((T*)s_data + s_id[3]); 
    
                // 初始预测值
                pred = tmp1;
    
                // 计算不同 case 对应的 pred
                if ( (case1 && case2 && case3) || (!case1 && case2 && case3 && case4)) {
                    pred = cubic_interpolator(tmp0, tmp1, tmp2, tmp3);
                    
                }
                else if ((case1 && case2 && !case3) || ( !case1 && case2 && !(case3 && case4) && case5)) {
                    pred = (-tmp0 + 6 * tmp1 + 3 * tmp2) / 8;
                }
                else if ((case1 && !case2 && case3) || (!case1 && !case2 && case3 && case4 )){
                    pred = (3 * tmp1 + 6 * tmp2 - tmp3) / 8;   
                }
                else if ((case1 && !case2 && !case3) || (!case1 && !case2 && !(case3 && case4) && case5)) {
                    pred = (tmp1 + tmp2) / 2;
                }

            }
            auto get_interp_order = [&](auto x, auto gx, auto gs){
                int b = (x >= 3 * unit) ? 3 : 1;
                int f = ((x + 3 * unit <= AnchorBlockSizeX * numAnchorBlockX) && ((gx + 3 * unit < gs))) ? 3 :
                (((gx + unit < gs)) ? 1 : 0);

                return (b == 3) ? ((f == 3) ? 4 : ((f == 1) ? 3 : 0)) 
                                : ((f == 3) ? 2 : ((f == 1) ? 1 : 0));
            };
            if CONSTEXPR (FACE) {  //

                bool I_YZ = (x % (2*unit) ) == 0;
                bool I_XZ = (y % (2*unit ) )== 0;
                int x_1,BI_1,GD_1,gx_1,gs_1;
                int x_2,BI_2,GD_2,gx_2,gs_2;
                int s_id_1[4], s_id_2[4];
                auto x_size = AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1);
                auto y_size = AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2);
                // auto z_size = AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3);
                if (I_YZ){
                   
                 x_1 = z,BI_1 = BIZ, GD_1 = GDZ, gx_1 = global_z,gs_1 = data_size.z;
                 x_2 = y,BI_2 = BIY, GD_2 = GDY, gx_2 = global_y, gs_2 = data_size.y;
                 s_id_1[0] = x_size * y_size * id_z[0] + x_size * y + x;
                 s_id_1[1] = x_size * y_size * id_z[1] + x_size * y + x;
                 s_id_1[2] = x_size * y_size * id_z[2] + x_size * y + x;
                 s_id_1[3] = x_size * y_size * id_z[3] + x_size * y + x;
                 s_id_2[0] = x_size * y_size * z + x_size * id_y[0] + x;
                 s_id_2[1] = x_size * y_size * z + x_size * id_y[1] + x;
                 s_id_2[2] = x_size * y_size * z + x_size * id_y[2] + x;
                 s_id_2[3] = x_size * y_size * z + x_size * id_y[3] + x;
                 pred = s_data[id_z[1]][id_y[1]][x];

                }
                else if (I_XZ){
                    x_1 = z,BI_1 = BIZ, GD_1 = GDZ, gx_1 = global_z,gs_1 = data_size.z;
                    x_2 = x,BI_2 = BIX, GD_2 = GDX, gx_2 = global_x, gs_2 = data_size.x;
                    s_id_1[0] = x_size * y_size * id_z[0] + x_size * y + x;
                    s_id_1[1] = x_size * y_size * id_z[1] + x_size * y + x;
                    s_id_1[2] = x_size * y_size * id_z[2] + x_size * y + x;
                    s_id_1[3] = x_size * y_size * id_z[3] + x_size * y + x;
                    
                    s_id_2[0] = x_size * y_size * z + x_size * y + id_x[0];
                    s_id_2[1] = x_size * y_size * z + x_size * y + id_x[1];
                    s_id_2[2] = x_size * y_size * z + x_size * y + id_x[2];
                    s_id_2[3] = x_size * y_size * z + x_size * y + id_x[3];
                    pred = s_data[id_z[1]][y][id_x[1]];
                    
                }
                else{
                    x_1 = y,BI_1 = BIY, GD_1 = GDY, gx_1 = global_y, gs_1 = data_size.y;
                    x_2 = x,BI_2 = BIX, GD_2 = GDX, gx_2 = global_x, gs_2 = data_size.x;
                    s_id_1[0] = x_size * y_size * z + x_size * id_y[0] + x;
                    s_id_1[1] = x_size * y_size * z + x_size * id_y[1] + x;
                    s_id_1[2] = x_size * y_size * z + x_size * id_y[2] + x;
                    s_id_1[3] = x_size * y_size * z + x_size * id_y[3] + x;
                    s_id_2[0] = x_size * y_size * z + x_size * y + id_x[0];
                    s_id_2[1] = x_size * y_size * z + x_size * y + id_x[1];
                    s_id_2[2] = x_size * y_size * z + x_size * y + id_x[2];
                    s_id_2[3] = x_size * y_size * z + x_size * y + id_x[3];
                    pred = s_data[z][id_y[1]][id_x[1]];
                }

                    auto interp_1 = get_interp_order(x_1,gx_1,gs_1);
                    auto interp_2 = get_interp_order(x_2,gx_2,gs_2);

                    int case_num = interp_1 + interp_2 * 5;

                    if (interp_1 == 4 && interp_2 == 4) {
                        pred = (cubic_interpolator(*((T*)s_data + s_id_1[0]), 
                        *((T*)s_data + s_id_1[1]), 
                        *((T*)s_data + s_id_1[2]), 
                        *((T*)s_data + s_id_1[3])) +
                         cubic_interpolator(*((T*)s_data + s_id_2[0]), 
                        *((T*)s_data + s_id_2[1]), 
                        *((T*)s_data + s_id_2[2]), 
                        *((T*)s_data + s_id_2[3]))) / 2;
                    } else if (interp_1 != 4 && interp_2 == 4) {
                        pred = cubic_interpolator(*((T*)s_data + s_id_2[0]), 
                        *((T*)s_data + s_id_2[1]), 
                        *((T*)s_data + s_id_2[2]), 
                        *((T*)s_data + s_id_2[3]));
                    } else if (interp_1 == 4 && interp_2 != 4) {
                        pred = cubic_interpolator(*((T*)s_data + s_id_1[0]), 
                        *((T*)s_data + s_id_1[1]), 
                        *((T*)s_data + s_id_1[2]), 
                        *((T*)s_data + s_id_1[3]));
                    } else if (interp_1 == 3 && interp_2 == 3) {
                        pred = (-(*((T*)s_data + s_id_2[0]))+6*(*((T*)s_data + s_id_2[1])) + 3*(*((T*)s_data + s_id_2[2]))) / 8;
                        pred += (-(*((T*)s_data + s_id_1[0]))+6*(*((T*)s_data + s_id_1[1])) + 3*(*((T*)s_data + s_id_1[2]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 3 && interp_2 == 2) {
                        pred = (3*(*((T*)s_data + s_id_2[1]))+6*(*((T*)s_data + s_id_2[2])) - (*((T*)s_data + s_id_2[3]))) / 8;
                        pred += (-(*((T*)s_data + s_id_1[0]))+6*(*((T*)s_data + s_id_1[1])) + 3*(*((T*)s_data + s_id_1[2]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 3 && interp_2 < 2) {
                        pred = (-(*((T*)s_data + s_id_1[0]))+6*(*((T*)s_data + s_id_1[1])) + 3*(*((T*)s_data + s_id_1[2]))) / 8;
                    } else if (interp_1 == 2 && interp_2 == 3) {
                        pred = (3*(*((T*)s_data + s_id_1[1]))+6*(*((T*)s_data + s_id_1[2])) - (*((T*)s_data + s_id_1[3]))) / 8;
                        pred += (-(*((T*)s_data + s_id_2[0]))+6*(*((T*)s_data + s_id_2[1])) + 3*(*((T*)s_data + s_id_2[2]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 2 && interp_2 == 2) {
                        pred = (3*(*((T*)s_data + s_id_1[1]))+6*(*((T*)s_data + s_id_1[2])) - (*((T*)s_data + s_id_1[3]))) / 8;
                        pred += (3*(*((T*)s_data + s_id_2[1]))+6*(*((T*)s_data + s_id_2[2])) - (*((T*)s_data + s_id_2[3]))) / 8;
                        pred /= 2;
                    } else if (interp_1 == 2 && interp_2 < 2) {
                        pred = (3*(*((T*)s_data + s_id_1[1]))+6*(*((T*)s_data + s_id_1[2])) - (*((T*)s_data + s_id_1[3]))) / 8;
                    } else if (interp_1 <= 1 && interp_2 == 3) {
                        pred = (-(*((T*)s_data + s_id_2[0]))+6*(*((T*)s_data + s_id_2[1])) + 3*(*((T*)s_data + s_id_2[2]))) / 8;
                    } else if (interp_1 <= 1 && interp_2 == 2) {
                        pred = (3*(*((T*)s_data + s_id_2[1]))+6*(*((T*)s_data + s_id_2[2])) - (*((T*)s_data + s_id_2[3]))) / 8;
                    } else if (interp_1 == 1 && interp_2 == 1) {
                        pred = ((*((T*)s_data + s_id_2[1]))+(*((T*)s_data + s_id_2[2]))) / 2;
                        pred += ((*((T*)s_data + s_id_1[1]))+(*((T*)s_data + s_id_1[2]))) / 2;
                        pred /= 2;
                    } else if (interp_1 == 1 && interp_2 < 1) {
                        
                        pred = ((*((T*)s_data + s_id_1[1]))+(*((T*)s_data + s_id_1[2]))) / 2;
                    } else if (interp_1 == 0 && interp_2 == 1) {
                        pred = ((*((T*)s_data + s_id_2[1]))+(*((T*)s_data + s_id_2[2]))) / 2;
                    }
                    else{
                        pred = (*((T*)s_data + s_id_1[1])) + (*((T*)s_data + s_id_2[1])) - pred;
                    }
                    
            }

            if CONSTEXPR (CUBE) {  //
                T tmp_z[4], tmp_y[4], tmp_x[4];
                auto interp_z = get_interp_order(z,global_z,data_size.z);
                auto interp_y = get_interp_order(y,global_y,data_size.y);
                auto interp_x = get_interp_order(x,global_x,data_size.x);
                
                #pragma unroll
                for(int id_itr = 0; id_itr < 4; ++id_itr){
                 tmp_x[id_itr] = s_data[z][y][id_x[id_itr]]; 
                }
                if(interp_z == 4){
                    #pragma unroll
                    for(int id_itr = 0; id_itr < 4; ++id_itr){
                        tmp_z[id_itr] = s_data[id_z[id_itr]][y][x];
                       }
                }
                if(interp_y == 4){
                    #pragma unroll
                    for(int id_itr = 0; id_itr < 4; ++id_itr){
                     tmp_y[id_itr] = s_data[z][id_y[id_itr]][x]; 
                    }
                }


                T pred_z[5], pred_y[5], pred_x[5];
                pred_x[0] = tmp_x[1];
                pred_x[1] = cubic_interpolator(tmp_x[0],tmp_x[1],tmp_x[2],tmp_x[3]);
                pred_x[2] = (-tmp_x[0]+6*tmp_x[1] + 3*tmp_x[2]) / 8;
                pred_x[3] = (3*tmp_x[1] + 6*tmp_x[2]-tmp_x[3]) / 8;
                pred_x[4] = (tmp_x[1] + tmp_x[2]) / 2;
                
                pred_y[1] = cubic_interpolator(tmp_y[0],tmp_y[1],tmp_y[2],tmp_y[3]);

                
                pred_z[1] = cubic_interpolator(tmp_z[0],tmp_z[1],tmp_z[2],tmp_z[3]);
                
            
                
                pred = pred_x[0];
                pred = (interp_z == 4 && interp_y == 4 && interp_x == 4) ? (pred_x[1] +  pred_y[1] + pred_z[1]) / 3 : pred;
                
                pred = (interp_z == 4 && interp_y == 4 && interp_x != 4) ? (pred_z[1] + pred_y[1]) / 2 : pred;
                pred = (interp_z == 4 && interp_y != 4 && interp_x == 4) ? (pred_z[1] + pred_x[1]) / 2 : pred;
                pred = (interp_z != 4 && interp_y == 4 && interp_x == 4) ? (pred_y[1] + pred_x[1]) / 2 : pred;
                
                pred = (interp_z == 4 && interp_y != 4 && interp_x != 4) ? pred_z[1]: pred;
                pred = (interp_z != 4 && interp_y == 4 && interp_x != 4) ? pred_y[1]: pred;
                pred = (interp_z != 4 && interp_y != 4 && interp_x == 4) ? pred_x[1]: pred;


                pred = (interp_z != 4 && interp_y != 4 && interp_x == 3) ? pred_x[2]: pred;
                pred = (interp_z != 4 && interp_y != 4 && interp_x == 2) ? pred_x[3]: pred;
                pred = (interp_z != 4 && interp_y != 4 && interp_x == 1) ? pred_x[4]: pred;
                // pred = (interp_z != 4 && interp_y != 4 && interp_x == 0) ? pred_x[0]: pred;
            }


  
            if CONSTEXPR (WORKFLOW == SPLINE3_AB_ATT) {
                
                auto          err = s_data[z][y][x] - pred;
                decltype(err) code;
                // TODO unsafe, did not deal with the out-of-cap case
                {
                    code = fabs(err) * eb_r + 1;
                    code = err < 0 ? -code : code;
                    code = int(code / 2) ;
                }
                s_data[z][y][x]  = pred + code * ebx2;
                atomicAdd(const_cast<T*>(error),code!=0);
            }
            else{
                atomicAdd(const_cast<T*>(error),fabs(s_data[z][y][x]-pred));
            }
        }
    };
    // -------------------------------------------------------------------------------- //

    if CONSTEXPR (COARSEN) {
        auto TOTAL = NUM_ELE;
            for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
                auto [x,y,z]    = xyzmap(_tix, unit);
                run(x, y, z);
            }   
    }
    else {
        if(TIX<NUM_ELE){
            auto [x,y,z]    = xyzmap(TIX, unit);
            run(x, y, z);
        }
    }
    __syncthreads();
}


template <typename T, typename FP, int LEVEL, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ, int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ, int LINEAR_BLOCK_SIZE, bool WORKFLOW>
__device__ void cusz::device_api::spline_layout_interpolate_att(
    volatile T s_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
    [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
    [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)],
    DIM3    data_size,
    DIM3 global_starts, FP eb_r, FP ebx2, uint8_t level, INTERPOLATION_PARAMS intp_param, volatile T *error)
{
    auto xblue = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2); };
    auto yblue = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
    auto zblue = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2 + 1); };

    auto xblue_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix ); };
    auto yblue_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy ); };
    auto zblue_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2 + 1); };

    auto xyellow = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2); };
    auto yyellow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2+1); };
    auto zyellow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };

    auto xyellow_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix ); };
    auto yyellow_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2 + 1); };
    auto zyellow_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2); };


    auto xhollow = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2 + 1); };
    auto yhollow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy); };
    auto zhollow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };

    auto xhollow_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2 + 1); };
    auto yhollow_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
    auto zhollow_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz *2); };

    auto nan_cubic_interp = [] __device__ (T a, T b, T c, T d) -> T{
        return (-a+9*b+9*c-d) / 16;
    };

    auto nat_cubic_interp = [] __device__ (T a, T b, T c, T d) -> T{
        return (-3*a+23*b+23*c-3*d) / 40;
    };
    constexpr auto COARSEN          = true;
    // constexpr auto NO_COARSEN       = false;
    constexpr auto BORDER_INCLUSIVE = true;
    // constexpr auto BORDER_EXCLUSIVE = false;


    int unit;

    FP cur_ebx2 = ebx2, cur_eb_r = eb_r;


    auto calc_eb = [&](auto unit) {
        cur_ebx2 = ebx2, cur_eb_r = eb_r;
        int temp = 1;
        while(temp < unit){
            temp *= 2;
            cur_eb_r *= intp_param.alpha;
            cur_ebx2 /= intp_param.alpha;
        }
        if(cur_ebx2 < ebx2 / intp_param.beta){
            cur_ebx2 = ebx2 / intp_param.beta;
            cur_eb_r = eb_r * intp_param.beta;
        }
    };

   
 
    
    if CONSTEXPR (WORKFLOW == SPLINE3_AB_ATT){
        int max_unit = ((AnchorBlockSizeX >= AnchorBlockSizeY) ? AnchorBlockSizeX : AnchorBlockSizeY);
        max_unit = ((max_unit >= AnchorBlockSizeZ) ? max_unit : AnchorBlockSizeZ);
        max_unit /= 2;
        int unit_x = AnchorBlockSizeX, unit_y = AnchorBlockSizeY, unit_z = AnchorBlockSizeZ;
        #pragma unroll
        for(int unit = max_unit; unit >= 1; unit /= 2){
            calc_eb(unit);
            unit_x = (SPLINE_DIM >= 1) ? unit * 2 : 1;
            unit_y = (SPLINE_DIM >= 2) ? unit * 2 : 1;
            unit_z = (SPLINE_DIM >= 3) ? unit * 2 : 1;
            if(intp_param.use_md[level]){
                int N_x = AnchorBlockSizeX / (unit * 2);
                int N_y = AnchorBlockSizeY / (unit * 2);
                int N_z = AnchorBlockSizeZ / (unit * 2);
                int N_line = N_x * (N_y + 1) * (N_z + 1) + (N_x + 1) * N_y * (N_z + 1) + (N_x + 1) * (N_y + 1) * N_z;
                int N_face = N_x * N_y * (N_z + 1) + N_x * (N_y + 1) * N_z + (N_x + 1) * N_y * N_z; 
                int N_cube = N_x * N_y * N_z;
                if(intp_param.use_natural[level] == 0){
                    if CONSTEXPR (SPLINE_DIM >= 1)
                    interpolate_stage_md_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>), true, false, false, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data,data_size,global_starts, xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r,cur_ebx2, nan_cubic_interp,error, N_line);

                    if CONSTEXPR (SPLINE_DIM >= 2)
                    interpolate_stage_md_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>), false, true, false, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, data_size,global_starts, xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r,cur_ebx2, nan_cubic_interp,error, N_face);

                    if CONSTEXPR (SPLINE_DIM >= 3)
                    interpolate_stage_md_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>), false, false, true, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data,data_size,global_starts, xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r,cur_ebx2, nan_cubic_interp, error, N_cube);
                }
                else{
                    if CONSTEXPR (SPLINE_DIM >= 1)
                    interpolate_stage_md_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>), true, false, false, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, data_size, global_starts, xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r,cur_ebx2, nat_cubic_interp, error, N_line);

                    if CONSTEXPR (SPLINE_DIM >= 2)
                    interpolate_stage_md_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>), false, true, false, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, data_size, global_starts, xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, nat_cubic_interp, error, N_face);

                    if CONSTEXPR (SPLINE_DIM >= 3)
                    interpolate_stage_md_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>), false, false, true, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, data_size, global_starts, xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, nat_cubic_interp, error, N_cube);
                }

            }
            else{
                if(intp_param.reverse[level]){
                    if CONSTEXPR (SPLINE_DIM >= 1){
                        interpolate_stage_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse),  false, false, true, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data, data_size, global_starts, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit, cur_eb_r,cur_ebx2, intp_param.use_natural[level], error, numAnchorBlockX * AnchorBlockSizeX / unit_x, numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                        unit_x /= 2;
                    }
                    if CONSTEXPR (SPLINE_DIM >= 2){
                        interpolate_stage_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse), false, true, false, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data, data_size, global_starts, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit, cur_eb_r,cur_ebx2, intp_param.use_natural[level], error, numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y, numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                        unit_y /= 2;
                    }
                    if CONSTEXPR (SPLINE_DIM >= 3){
                        interpolate_stage_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse), true, false, false, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data,data_size,global_starts, xblue_reverse, yblue_reverse, zblue_reverse, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[level],error, numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z);
                        unit_z /= 2;
                    }
                }
                else{
                    if CONSTEXPR (SPLINE_DIM >= 3){
                        interpolate_stage_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xblue), decltype(yblue), decltype(zblue),  true, false, false, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data,data_size,global_starts, xblue, yblue, zblue, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[level], error, numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z);
                        unit_z /= 2;
                    }
                    if CONSTEXPR (SPLINE_DIM >= 2){
                        interpolate_stage_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyellow), decltype(yyellow), decltype(zyellow), false, true, false, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data,data_size,global_starts, xyellow, yyellow, zyellow, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[level],error, numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y, numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                        unit_y /= 2;
                    }
                    if CONSTEXPR (SPLINE_DIM >= 1){
                        interpolate_stage_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xhollow), decltype(yhollow), decltype(zhollow), false, false, true, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data,data_size,global_starts, xhollow, yhollow, zhollow, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[level],error,  numAnchorBlockX * AnchorBlockSizeX / unit_x, numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                        unit_x /= 2;
                    }
                }
            }
        }
    }

    if CONSTEXPR (WORKFLOW != SPLINE3_AB_ATT){
        
        unit = 1 << level;
        int unit_x = (SPLINE_DIM >= 1) ? unit * 2 : 1;
        int unit_y = (SPLINE_DIM >= 2) ? unit * 2 : 1;
        int unit_z = (SPLINE_DIM >= 3) ? unit * 2 : 1;
        if(intp_param.use_md[level]){
            int N_x = AnchorBlockSizeX / (unit * 2);
            int N_y = AnchorBlockSizeY / (unit * 2);
            int N_z = AnchorBlockSizeZ / (unit * 2);
            int N_line = N_x * (N_y + 1) * (N_z + 1) + (N_x + 1) * N_y * (N_z + 1) + (N_x + 1) * (N_y + 1) * N_z;
            int N_face = N_x * N_y * (N_z + 1) + N_x * (N_y + 1) * N_z + (N_x + 1) * N_y * N_z; 
            int N_cube = N_x * N_y * N_z;

            // auto cubic_interp = (intp_param.use_natural[level] == 0) ? nan_cubic_interp : nat_cubic_interp;

            if(intp_param.use_natural[level] == 0){
                if CONSTEXPR (SPLINE_DIM >= 1)
                interpolate_stage_md_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>), true, false, false, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data,data_size,global_starts, xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r,cur_ebx2, nan_cubic_interp,error, N_line);

                if CONSTEXPR (SPLINE_DIM >= 2)
                interpolate_stage_md_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>), false, true, false, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, data_size,global_starts, xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r,cur_ebx2, nan_cubic_interp,error, N_face);

                if CONSTEXPR (SPLINE_DIM >= 3)
                interpolate_stage_md_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>), false, false, true, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data,data_size,global_starts, xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r,cur_ebx2, nan_cubic_interp, error, N_cube);
            }
            else{
                if CONSTEXPR (SPLINE_DIM >= 1)
                interpolate_stage_md_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>), true, false, false, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, data_size, global_starts, xyzmap_line<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r,cur_ebx2, nat_cubic_interp, error, N_line);

                if CONSTEXPR (SPLINE_DIM >= 2)
                interpolate_stage_md_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>), false, true, false, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, data_size, global_starts, xyzmap_face<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, nat_cubic_interp, error, N_face);

                if CONSTEXPR (SPLINE_DIM >= 3)
                interpolate_stage_md_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>), false, false, true, LINEAR_BLOCK_SIZE, COARSEN, BORDER_INCLUSIVE, WORKFLOW>(s_data, data_size, global_starts, xyzmap_cube<SPLINE_DIM, AnchorBlockSizeX>, unit, cur_eb_r, cur_ebx2, nat_cubic_interp, error, N_cube);
            }

        }
        else{
            if(intp_param.reverse[level]){
                if CONSTEXPR (SPLINE_DIM >= 1){
                    interpolate_stage_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse),  false, false, true, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data, data_size, global_starts, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit, cur_eb_r,cur_ebx2, intp_param.use_natural[level], error, numAnchorBlockX * AnchorBlockSizeX / unit_x, numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                    unit_x /= 2;
                }
                if CONSTEXPR (SPLINE_DIM >= 2){
                    interpolate_stage_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse), false, true, false, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data, data_size, global_starts, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit, cur_eb_r,cur_ebx2, intp_param.use_natural[level], error, numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y, numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                    unit_y /= 2;
                }
                if CONSTEXPR (SPLINE_DIM >= 3){
                    interpolate_stage_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse), true, false, false, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data,data_size,global_starts, xblue_reverse, yblue_reverse, zblue_reverse, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[level],error, numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z);
                    unit_z /= 2;
                }
            }
            else{
                if CONSTEXPR (SPLINE_DIM >= 3){
                    interpolate_stage_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xblue), decltype(yblue), decltype(zblue),  true, false, false, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data,data_size,global_starts, xblue, yblue, zblue, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[level], error, numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z);
                    unit_z /= 2;
                }
                if CONSTEXPR (SPLINE_DIM >= 2){
                    interpolate_stage_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xyellow), decltype(yyellow), decltype(zyellow), false, true, false, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data,data_size,global_starts, xyellow, yyellow, zyellow, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[level],error, numAnchorBlockX * AnchorBlockSizeX / unit_x + (SPLINE_DIM >= 1), numAnchorBlockY * AnchorBlockSizeY / unit_y, numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                    unit_y /= 2;
                }
                if CONSTEXPR (SPLINE_DIM >= 1){
                    interpolate_stage_att<T, FP, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, decltype(xhollow), decltype(yhollow), decltype(zhollow), false, false, true, COARSEN, LINEAR_BLOCK_SIZE, BORDER_INCLUSIVE, WORKFLOW>(s_data,data_size,global_starts, xhollow, yhollow, zhollow, unit,cur_eb_r,cur_ebx2, intp_param.use_natural[level],error,  numAnchorBlockX * AnchorBlockSizeX / unit_x, numAnchorBlockY * AnchorBlockSizeY / unit_y + (SPLINE_DIM >= 2), numAnchorBlockZ * AnchorBlockSizeZ / unit_z + (SPLINE_DIM >= 3));
                    unit_x /= 2;
                }
            }
        }
    }
}



template <typename TITER, typename FP, int LEVEL, int SPLINE_DIM, int AnchorBlockSizeX, int AnchorBlockSizeY, int AnchorBlockSizeZ, int numAnchorBlockX, int numAnchorBlockY, int numAnchorBlockZ, int LINEAR_BLOCK_SIZE>
__global__ void cusz::pa_spline_infprecis_data(
    TITER   data,
    DIM3    data_size,
    STRIDE3 data_leap,
    DIM3 sample_starts,
    DIM3 sample_block_grid_sizes,
    DIM3 sample_strides,
    FP eb_r,
    FP eb_x2,
    INTERPOLATION_PARAMS intp_param,
    TITER errors,
    bool workflow
    )
{
    // compile time variables
    using T = typename std::remove_pointer<TITER>::type;

    {
        // if CONSTEXPR (SPLINE_DIM == 3)
        // __shared__ struct {
        //     T data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
        //     [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
        //     [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
        //     T err[6];
        // } shmem;
        
        __shared__    T shmem_data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
        [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
        [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
        __shared__    T shmem_err[9];
        
        // if CONSTEXPR (SPLINE_DIM == 2)
        // __shared__ struct {
        //     T data[AnchorBlockSizeZ * numAnchorBlockZ + (SPLINE_DIM >= 3)]
        //     [AnchorBlockSizeY * numAnchorBlockY + (SPLINE_DIM >= 2)]
        //     [AnchorBlockSizeX * numAnchorBlockX + (SPLINE_DIM >= 1)];
        //     T err[1];
        // } shmem;

        DIM3 global_starts;
        uint8_t level = 0;
        uint8_t unit = 1;
        pre_compute_att<T, SPLINE_DIM, LEVEL>(sample_starts, sample_block_grid_sizes, sample_strides, global_starts, intp_param, level, unit, shmem_err, workflow);
        
        global2shmem_data_att<T, T, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ,LINEAR_BLOCK_SIZE>(data, data_size, data_leap, shmem_data,global_starts,unit);
        
        if CONSTEXPR (SPLINE_DIM == 3){
            if(workflow){
                if(level==2){
                    uint8_t level3 = 3;
                    intp_param.use_natural[3] = false;
                    intp_param.use_natural[2] = false;
                    intp_param.use_md[3] = false;
                    intp_param.reverse[3] = false;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level3,intp_param,shmem_err);
                    intp_param.reverse[3] = true;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level3,intp_param,shmem_err+1);
                    intp_param.use_md[3] = true;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level3,intp_param,shmem_err+2);


                    intp_param.use_md[2] = false;
                    intp_param.reverse[2] = false;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem_err+3);
                    intp_param.reverse[2] = true;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem_err+4);
                    intp_param.use_md[2] = true;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem_err+5);
                    if(TIX<6){
                        atomicAdd(const_cast<T*>(errors+TIX),shmem_err[TIX]);
                    }
                }
                else if (level == 1){
                    intp_param.use_md[1] = false;
                    intp_param.reverse[1] = false;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem_err);
                    intp_param.reverse[1] = true;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem_err+1);
                    intp_param.use_md[1] = true;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem_err+2);

                    if(TIX<3){
                    atomicAdd(const_cast<T*>(errors + 3 + BIY * 3 + TIX),shmem_err[TIX]);
                    }
                }
                else{
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size, global_starts, eb_r, eb_x2, level, intp_param, shmem_err);
                    if(TIX==0){
                        atomicAdd(const_cast<T*>(errors + 9 + BIY), shmem_err[0]);
                    }
                }
                
            }
            else{
                cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_AB_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem_err);
                if(TIX==0)
                    atomicAdd(const_cast<T*>(errors+BIY),shmem_err[0]);
            
            }
        }
        if CONSTEXPR (SPLINE_DIM == 2){
            if(workflow){
                if(level==3){
                    uint8_t level5 = 5;
                    intp_param.use_natural[5] = false;
                    intp_param.use_natural[4] = false;
                    intp_param.use_natural[3] = false;
                    intp_param.use_md[5] = false;
                    intp_param.reverse[5] = false;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level5,intp_param,shmem_err);
                    intp_param.reverse[5] = true;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level5,intp_param,shmem_err+1);
                    intp_param.use_md[5] = true;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level5,intp_param,shmem_err+2);

                    uint8_t level4 = 4;
                    intp_param.use_md[4] = false;
                    intp_param.reverse[4] = false;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level4,intp_param,shmem_err+3);
                    intp_param.reverse[4] = true;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level4,intp_param,shmem_err+4);
                    intp_param.use_md[4] = true;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level4,intp_param,shmem_err+5);
                   
                    intp_param.use_md[3] = false;
                    intp_param.reverse[3] = false;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem_err+6);
                    intp_param.reverse[3] = true;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem_err+7);
                    intp_param.use_md[3] = true;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem_err+8);
                    if(TIX<9){
                        atomicAdd(const_cast<T*>(errors+TIX),shmem_err[TIX]);
                    }
                }
                else if (level == 2){
                    intp_param.use_md[2] = false;
                    intp_param.reverse[2] = false;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem_err);
                    intp_param.reverse[2] = true;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem_err+1);
                    intp_param.use_md[2] = true;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem_err+2);

                    if(TIX<3){
                    atomicAdd(const_cast<T*>(errors + 6 + BIY * 3 + TIX),shmem_err[TIX]);
                    }
                }
                else if (level == 1){
                    intp_param.use_md[1] = false;
                    intp_param.reverse[1] = false;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem_err);
                    intp_param.reverse[1] = true;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem_err+1);
                    intp_param.use_md[1] = true;
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem_err+2);

                    if(TIX<3){
                    atomicAdd(const_cast<T*>(errors + 6 + BIY * 3 + TIX),shmem_err[TIX]);
                    }
                }
                else{
                    cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_PRED_ATT>(shmem_data, data_size, global_starts, eb_r, eb_x2, level, intp_param, shmem_err);
                    if(TIX==0){
                        atomicAdd(const_cast<T*>(errors + 15 + BIY), shmem_err[0]);
                    }
                }
                
            }
            else{
                cusz::device_api::spline_layout_interpolate_att<T, FP, LEVEL, SPLINE_DIM, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, LINEAR_BLOCK_SIZE, SPLINE3_AB_ATT>(shmem_data, data_size,global_starts,eb_r,eb_x2,level,intp_param,shmem_err);
                if(TIX==0)
                    atomicAdd(const_cast<T*>(errors+BIY),shmem_err[0]);
            
            }
        }
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
#undef GDX
#undef GDY
#undef GDZ

#endif