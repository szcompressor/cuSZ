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
#include "cusz/type.h"
#include "utils/err.hh"
#include "utils/timer.hh"

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
#define BDZ blockDim.z
#define GDX gridDim.x
#define GDY gridDim.y
#define GDZ gridDim.z

using DIM     = u4;
using STRIDE  = u4;
using DIM3    = dim3;
using STRIDE3 = dim3;

constexpr int BLOCK8  = 8;
constexpr int BLOCK32 = 32;
constexpr int DEFAULT_LINEAR_BLOCK_SIZE = 384;

#define SHM_ERROR s_ectrl

namespace cusz {

/********************************************************************************
 * host API
 ********************************************************************************/

template <
    typename TITER,
    typename EITER,
    typename FP            = float,
    int  LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE, 
    typename CompactVal = TITER,
    typename CompactIdx = uint32_t*,
    typename CompactNum = uint32_t*>
__global__ void c_spline3d_infprecis_32x8x8data(
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
    INTERPOLATION_PARAMS intp_param);

template <
    typename EITER,
    typename TITER,
    typename FP           = float,
    int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
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
    int     radius,
    INTERPOLATION_PARAMS intp_param);

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
spline3d_layout2_interpolate(volatile T1 s_data[9][9][33], volatile T2 s_ectrl[9][9][33], FP eb_r, FP ebx2, int radius, INTERPOLATION_PARAMS intp_param);
}  // namespace device_api

}  // namespace cusz

/********************************************************************************
 * helper function
 ********************************************************************************/

namespace {

template <bool INCLUSIVE = true>
__forceinline__ __device__ bool xyz33x9x9_predicate(unsigned int x, unsigned int y, unsigned int z,const DIM3 &data_size)
{
    if CONSTEXPR (INCLUSIVE) {  //

        
            return (x <= 32 and y <= 8 and z <= 8) and BIX*BLOCK32+x<data_size.x and BIY*BLOCK8+y<data_size.y and BIZ*BLOCK8+z<data_size.z;
    }
    else {
        return x < 32+(BIX==GDX-1) and y < 8+(BIY==GDY-1) and z < 8+(BIZ==GDZ-1) and BIX*BLOCK32+x<data_size.x and BIY*BLOCK8+y<data_size.y and BIZ*BLOCK8+z<data_size.z;
    }
}

// control block_id3 in function call
template <typename T, bool PRINT_FP = true, int XEND = 33, int YEND = 9, int ZEND = 9>
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

template <typename T1, typename T2, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
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

template <typename T1, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
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
template <typename T1, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
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

template <typename T1, typename T2 = T1, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
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

template <typename T1, typename T2, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
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
/*
        if(BIX == 7 and BIY == 47 and BIZ == 15 and x==10 and y==8 and z==4){
            printf("g2s1084 %d %d %d %d %.2e %.2e \n",gx,gy,gz,gid,s_data[z][y][x],data[gid]);
        }

        if(BIX == 7 and BIY == 47 and BIZ == 15 and x==10 and y==4 and z==8){
            printf("g2s1048 %d %d %d %d %.2e %.2e \n",gx,gy,gz,gid,s_data[z][y][x],data[gid]);
        }*/
    }
    __syncthreads();
}


template <typename T = float, typename E = u4, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void global2shmem_fuse(E* ectrl, dim3 ectrl_size, dim3 ectrl_leap, T* scattered_outlier, volatile T s_ectrl[9][9][33])
{
    constexpr auto TOTAL = 33 * 9 * 9;

    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
        auto x   = (_tix % 33);
        auto y   = (_tix / 33) % 9;
        auto z   = (_tix / 33) / 9;
        auto gx  = (x + BIX * BLOCK32);
        auto gy  = (y + BIY * BLOCK8);
        auto gz  = (z + BIZ * BLOCK8);
        auto gid = gx + gy * ectrl_leap.y + gz * ectrl_leap.z;

        if (gx < ectrl_size.x and gy < ectrl_size.y and gz < ectrl_size.z) s_ectrl[z][y][x] = static_cast<T>(ectrl[gid]) + scattered_outlier[gid];
    }
    __syncthreads();
}

// dram_outlier should be the same in type with shared memory buf
template <typename T1, typename T2, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void
shmem2global_32x8x8data(volatile T1 s_buf[9][9][33], T2* dram_buf, DIM3 buf_size, STRIDE3 buf_leap)
{
    auto x_size=BLOCK32+(BIX==GDX-1);
    auto y_size=BLOCK8+(BIY==GDY-1);
    auto z_size=BLOCK8+(BIZ==GDZ-1);
    //constexpr auto TOTAL = 32 * 8 * 8;
    auto TOTAL = x_size * y_size * z_size;

    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
        auto x   = (_tix % x_size);
        auto y   = (_tix / x_size) % y_size;
        auto z   = (_tix / x_size) / y_size;
        auto gx  = (x + BIX * BLOCK32);
        auto gy  = (y + BIY * BLOCK8);
        auto gz  = (z + BIZ * BLOCK8);
        auto gid = gx + gy * buf_leap.y + gz * buf_leap.z;

        if (gx < buf_size.x and gy < buf_size.y and gz < buf_size.z) dram_buf[gid] = s_buf[z][y][x];
        /*
        if(BIX == 7 and BIY == 47 and BIZ == 15 and x==10 and y==8 and z==4){
            printf("s2g1084 %d %d %d %d %.2e %.2e \n",gx,gy,gz,gid,s_buf[z][y][x],dram_buf[gid]);
        }

        if(BIX == 7 and BIY == 47 and BIZ == 15 and x==10 and y==4 and z==8){
            printf("s2g1048 %d %d %d %d %.2e %.2e \n",gx,gy,gz,gid,s_buf[z][y][x],dram_buf[gid]);
        }*/
    }
    __syncthreads();
}


// dram_outlier should be the same in type with shared memory buf
template <typename T1, typename T2, int LINEAR_BLOCK_SIZE = DEFAULT_LINEAR_BLOCK_SIZE>
__device__ void
shmem2global_32x8x8data_with_compaction(volatile T1 s_buf[9][9][33], T2* dram_buf, DIM3 buf_size, STRIDE3 buf_leap, int radius, T1* dram_compactval = nullptr, uint32_t* dram_compactidx = nullptr, uint32_t* dram_compactnum = nullptr)
{
    auto x_size = BLOCK32 + (BIX == GDX-1);
    auto y_size = BLOCK8 + (BIY == GDY-1);
    auto z_size = BLOCK8 + (BIZ == GDZ-1);
    auto TOTAL = x_size * y_size * z_size;

    for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
        auto x   = (_tix % x_size);
        auto y   = (_tix / x_size) % y_size;
        auto z   = (_tix / x_size) / y_size;
        auto gx  = (x + BIX * BLOCK32);
        auto gy  = (y + BIY * BLOCK8);
        auto gz  = (z + BIZ * BLOCK8);
        auto gid = gx + gy * buf_leap.y + gz * buf_leap.z;

        auto candidate = s_buf[z][y][x];
        bool quantizable = (candidate >= 0) and (candidate < 2*radius);

        if (gx < buf_size.x and gy < buf_size.y and gz < buf_size.z) {
            // TODO this is for algorithmic demo by reading from shmem
            // For performance purpose, it can be inlined in quantization
            dram_buf[gid] = quantizable * static_cast<T2>(candidate);

            if (not quantizable) {
                auto cur_idx = atomicAdd(dram_compactnum, 1);
                dram_compactidx[cur_idx] = gid;
                dram_compactval[cur_idx] = candidate;
            }
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
    volatile T1 s_data[9][9][33],
    volatile T2 s_ectrl[9][9][33],
    DIM3    data_size,
    LAMBDAX     xmap,
    LAMBDAY     ymap,
    LAMBDAZ     zmap,
    int         unit,
    FP          eb_r,
    FP          ebx2,
    int         radius,
    bool cubic)
{
    static_assert(BLOCK_DIMX * BLOCK_DIMY * (COARSEN ? 1 : BLOCK_DIMZ) <= 384, "block oversized");
    static_assert((BLUE or YELLOW or HOLLOW) == true, "must be one hot");
    static_assert((BLUE and YELLOW) == false, "must be only one hot (1)");
    static_assert((BLUE and YELLOW) == false, "must be only one hot (2)");
    static_assert((YELLOW and HOLLOW) == false, "must be only one hot (3)");

    auto run = [&](auto x, auto y, auto z) {

        

        if (xyz33x9x9_predicate<BORDER_INCLUSIVE>(x, y, z,data_size)) {
            T1 pred = 0;

            //if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and (CONSTEXPR (YELLOW)) )
            //    printf("%d %d %d\n",x,y,z);
            /*
             if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and x==4 and y==4 and z==4)
                        printf("444 %.2e %.2e \n",s_data[z - unit][y][x],s_data[z + unit][y][x]);

            if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and x==4 and y==4 and z==0)
                        printf("440 %.2e %.2e \n",s_data[z][y - unit][x],s_data[z][y + unit][x]);
            if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and x==4 and y==8 and z==0)
                        printf("480 %.2e %.2e \n",s_data[z][y ][x- unit],s_data[z][y ][x+ unit]);*/
                  //  }
            auto global_x=BIX*BLOCK32+x, global_y=BIY*BLOCK8+y, global_z=BIZ*BLOCK8+z;
            if(cubic){
                if CONSTEXPR (BLUE) {  //

                    if(BIZ!=GDZ-1){

                        if(z>=3*unit and z+3*unit<=BLOCK8  )
                            pred = (-s_data[z - 3*unit][y][x]+9*s_data[z - unit][y][x] + 9*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 16;
                        else if (z+3*unit<=BLOCK8)
                            pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 8;
                        else if (z>=3*unit)
                            pred = (-s_data[z - 3*unit][y][x]+6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]) / 8;

                        else
                            pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
                    }
                    else{
                        if(z>=3*unit){
                            if(z+3*unit<=BLOCK8 and global_z+3*unit<data_size.z)
                                pred = (-s_data[z - 3*unit][y][x]+9*s_data[z - unit][y][x] + 9*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 16;
                            else if (global_z+unit<data_size.z)
                                pred = (-s_data[z - 3*unit][y][x]+6*s_data[z - unit][y][x] + 3*s_data[z + unit][y][x]) / 8;
                            else
                                pred=s_data[z - unit][y][x];

                        }
                        else{
                            if(z+3*unit<=BLOCK8 and global_z+3*unit<data_size.z)
                                pred = (3*s_data[z - unit][y][x] + 6*s_data[z + unit][y][x]-s_data[z + 3*unit][y][x]) / 8;
                            else if (global_z+unit<data_size.z)
                                pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
                            else
                                pred=s_data[z - unit][y][x];
                        } 
                    }
                }
                if CONSTEXPR (YELLOW) {  //
                   // if(BIX == 5 and BIY == 22 and BIZ == 6 and unit==1 and x==29 and y==7 and z==0){
                   //     printf("%.2e %.2e %.2e %.2e\n",s_data[z ][y- 3*unit][x],s_data[z ][y- unit][x],s_data[z ][y+ unit][x]);
                  //  }
                    if(BIY!=GDY-1){
                        if(y>=3*unit and y+3*unit<=BLOCK8 )
                            pred = (-s_data[z ][y- 3*unit][x]+9*s_data[z ][y- unit][x] + 9*s_data[z ][y+ unit][x]-s_data[z][y + 3*unit][x]) / 16;
                        else if (y+3*unit<=BLOCK8)
                            pred = (3*s_data[z ][y - unit][x] + 6*s_data[z][y + unit][x]-s_data[z][y + 3*unit][x]) / 8;
                        else if (y>=3*unit)
                            pred = (-s_data[z ][y- 3*unit][x]+6*s_data[z][y - unit][x] + 3*s_data[z][y + unit][x]) / 8;
                        else
                            pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
                    }
                    else{
                        if(y>=3*unit){
                            if(y+3*unit<=BLOCK8 and global_y+3*unit<data_size.y)
                                pred = (-s_data[z ][y- 3*unit][x]+9*s_data[z][y - unit][x] + 9*s_data[z ][y+ unit][x]-s_data[z ][y+ 3*unit][x]) / 16;
                            else if (global_y+unit<data_size.y)
                                pred = (-s_data[z ][y- 3*unit][x]+6*s_data[z ][y- unit][x] + 3*s_data[z ][y+ unit][x]) / 8;
                            else
                                pred=s_data[z ][y- unit][x];

                        }
                        else{
                            if(y+3*unit<=BLOCK8 and global_y+3*unit<data_size.y)
                                pred = (3*s_data[z][y - unit][x] + 6*s_data[z ][y+ unit][x]-s_data[z][y + 3*unit][x]) / 8;
                            else if (global_y+unit<data_size.y)
                                pred = (s_data[z ][y- unit][x] + s_data[z][y + unit][x]) / 2;
                            else
                                pred=s_data[z ][y- unit][x];
                        } 
                    }
                }

                if CONSTEXPR (HOLLOW) {  //
                    //if(BIX == 5 and BIY == 22 and BIZ == 6 and unit==1)
                    //    printf("%d %d %d\n",x,y,z);
                    if(BIX!=GDX-1){
                        if(x>=3*unit and x+3*unit<=BLOCK32 )
                            pred = (-s_data[z ][y][x- 3*unit]+9*s_data[z ][y][x- unit] + 9*s_data[z ][y][x+ unit]-s_data[z ][y][x + 3*unit]) / 16;
                        else if (x+3*unit<=BLOCK32)
                            pred = (3*s_data[z ][y][x- unit] + 6*s_data[z ][y][x + unit]-s_data[z][y][x + 3*unit]) / 8;
                        else if (x>=3*unit)
                            pred = (-s_data[z][y][x - 3*unit]+6*s_data[z][y][x - unit] + 3*s_data[z ][y][x + unit]) / 8;
                        else
                            pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
                    }
                    else{
                        if(x>=3*unit){
                            if(x+3*unit<=BLOCK32 and global_x+3*unit<data_size.x)
                                pred = (-s_data[z ][y][x- 3*unit]+9*s_data[z][y ][x- unit] + 9*s_data[z ][y][x+ unit]-s_data[z ][y][x+ 3*unit]) / 16;
                            else if (global_x+unit<data_size.x)
                                pred = (-s_data[z ][y][x- 3*unit]+6*s_data[z ][y][x- unit] + 3*s_data[z ][y][x+ unit]) / 8;
                            else
                                pred=s_data[z ][y][x- unit];

                        }
                        else{
                            if(x+3*unit<=BLOCK32 and global_x+3*unit<data_size.x)
                                pred = (3*s_data[z][y ][x- unit] + 6*s_data[z ][y][x+ unit]-s_data[z][y ][x+ 3*unit]) / 8;
                            else if (global_x+unit<data_size.x)
                                pred = (s_data[z ][y][x- unit] + s_data[z][y ][x+ unit]) / 2;
                            else
                                pred=s_data[z ][y][x- unit];
                        } 
                    }
                }
                
            }
            else{
                if CONSTEXPR (BLUE) {  //
                    if(global_z+unit<data_size.z)
                    
                        pred = (s_data[z - unit][y][x] + s_data[z + unit][y][x]) / 2;
                    else
                        pred=s_data[z - unit][y][x];
                }
                if CONSTEXPR (YELLOW) {  //
                    if(global_y+unit<data_size.y)
                    
                        pred = (s_data[z][y - unit][x] + s_data[z][y + unit][x]) / 2;
                    else
                        pred = s_data[z][y - unit][x];
                }

                if CONSTEXPR (HOLLOW) {  //
                    if(global_x+unit<data_size.x)
                        pred = (s_data[z][y][x - unit] + s_data[z][y][x + unit]) / 2;
                    else
                        pred = s_data[z][y][x - unit];
                }
                
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
                /*
                  if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and x==4 and y==4 and z==0)
                        printf("440pred %.2e %.2e %.2e\n",pred,code,s_data[z][y][x]);
                    if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and x==4 and y==8 and z==0)
                        printf("480pred %.2e %.2e %.2e\n",pred,code,s_data[z][y][x]);
                        */
               // if(fabs(pred)>=3)
               //     printf("%d %d %d %d %d %d %d %d %d %d %.2e %.2e %.2e\n",unit,CONSTEXPR (BLUE),CONSTEXPR (YELLOW),CONSTEXPR (HOLLOW),BIX,BIY,BIZ,x,y,z,pred,code,s_data[z][y][x]);
              
                s_data[z][y][x]  = pred + (code - radius) * ebx2;
                

            }
            else {  // TODO == DECOMPRESSS and static_assert
                auto code       = s_ectrl[z][y][x];
                s_data[z][y][x] = pred + (code - radius) * ebx2;
                /*
                if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and x==4 and y==4 and z==0)
                        printf("440pred %.2e %.2e %.2e\n",pred,code,s_data[z][y][x]);
                    if(BIX == 7 and BIY == 47 and BIZ == 15 and unit==4 and x==4 and y==8 and z==0)
                        printf("480pred %.2e %.2e %.2e\n",pred,code,s_data[z][y][x]);
                        */

                //if(BIX == 4 and BIY == 20 and BIZ == 20 and unit==1 and CONSTEXPR (BLUE)){
               //     if(fabs(s_data[z][y][x])>=3)

              //      printf("%d %d %d %d %d %d %d %d %d %d %.2e %.2e %.2e\n",unit,CONSTEXPR (BLUE),CONSTEXPR (YELLOW),CONSTEXPR (HOLLOW),BIX,BIY,BIZ,x,y,z,pred,code,s_data[z][y][x]);
               // }
            }
        }
    };
    // -------------------------------------------------------------------------------- //

    if CONSTEXPR (COARSEN) {
        constexpr auto TOTAL = BLOCK_DIMX * BLOCK_DIMY * BLOCK_DIMZ;
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
        //}
        //may have bug    
        /*
        else{
            for (auto _tix = TIX; _tix < TOTAL; _tix += LINEAR_BLOCK_SIZE) {
                auto itix = (_tix % BLOCK_DIMX);
                auto itiz = (_tix / BLOCK_DIMX) % BLOCK_DIMZ;
                auto itiy = (_tix / BLOCK_DIMX) / BLOCK_DIMZ;
                auto x    = xmap(itix, unit);
                auto y    = ymap(itiy, unit);
                auto z    = zmap(itiz, unit);
                run(x, y, z);
            }
        }*/
        //may have bug  end
        
    }
    else {
        auto itix = (TIX % BLOCK_DIMX);
        auto itiy = (TIX / BLOCK_DIMX) % BLOCK_DIMY;
        auto itiz = (TIX / BLOCK_DIMX) / BLOCK_DIMY;
        auto x    = xmap(itix, unit);
        auto y    = ymap(itiy, unit);
        auto z    = zmap(itiz, unit);

        

     //   printf("%d %d %d\n", x,y,z);
        run(x, y, z);
    }
    __syncthreads();
}

}  // namespace

/********************************************************************************/

template <typename T1, typename T2, typename FP,int LINEAR_BLOCK_SIZE, bool WORKFLOW, bool PROBE_PRED_ERROR>
__device__ void cusz::device_api::spline3d_layout2_interpolate(
    volatile T1 s_data[9][9][33],
    volatile T2 s_ectrl[9][9][33],
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

    constexpr auto COARSEN          = true;
    constexpr auto NO_COARSEN       = false;
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

    // iteration 1
    /*
    auto colors_0={false,false,false};
    auto colors_1={false,false,false};
    auto colors_2={false,false,false};
    //auto interp_orders={0,1,2};

    auto set_orders [&](auto reverse){
        colors_0={false,false,false};
        colors_1={false,false,false};
        colors_2={false,false,false};
        auto interp_orders={0,1,2};
        if (reverse)
            interp_orders={2,1,0};
        colors_0[interp_orders[0]]=true;
        colors_1[interp_orders[1]]=true;
        colors_2[interp_orders[2]]=true;

    }
    */
    


    int unit = 4;
    calc_eb(unit);
    //set_orders(reverse[2]);
    if(intp_param.reverse[2]){
        interpolate_stage<
            T1, T2, FP, decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse),  //
            false, false, true, LINEAR_BLOCK_SIZE, 4, 2, NO_COARSEN, 2, BORDER_INCLUSIVE, WORKFLOW>(
            s_data, s_ectrl,data_size, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[2]);

        interpolate_stage<
            T1, T2, FP, decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse),  //
            false, true, false, LINEAR_BLOCK_SIZE, 9, 1, NO_COARSEN, 2, BORDER_INCLUSIVE, WORKFLOW>(
            s_data, s_ectrl,data_size, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[2]);
        interpolate_stage<
            T1, T2, FP, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse),  //
            true, false, false, LINEAR_BLOCK_SIZE, 9, 3, NO_COARSEN, 1, BORDER_INCLUSIVE, WORKFLOW>(
            s_data, s_ectrl,data_size, xblue_reverse, yblue_reverse, zblue_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[2]);


    }
    else{
        //if( BIX==0 and BIY==0 and BIZ==0)
       // printf("lv3s0\n");
        interpolate_stage<
            T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
            true, false, false, LINEAR_BLOCK_SIZE, 5, 2, NO_COARSEN, 1, BORDER_INCLUSIVE, WORKFLOW>(
            s_data, s_ectrl,data_size, xblue, yblue, zblue, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[2]);
       // if(BIX==0 and BIY==0 and BIZ==0)
       // printf("lv3s1\n");
        interpolate_stage<
            T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
            false, true, false, LINEAR_BLOCK_SIZE, 5, 1, NO_COARSEN, 3, BORDER_INCLUSIVE, WORKFLOW>(
            s_data, s_ectrl,data_size, xyellow, yyellow, zyellow, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[2]);
        //if(BIX==0 and BIY==0 and BIZ==0)
      //  printf("lv3s2\n");
        interpolate_stage<
            T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
            false, false, true, LINEAR_BLOCK_SIZE, 4, 3, NO_COARSEN, 3, BORDER_INCLUSIVE, WORKFLOW>(
            s_data, s_ectrl,data_size, xhollow, yhollow, zhollow, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[2]);
    }
   // if(BIX==0 and BIY==0 and BIZ==0)
   // printf("lv3\n");

    unit = 2;
    calc_eb(unit);
    //set_orders(reverse[1]);

    // iteration 2, TODO switch y-z order
    if(intp_param.reverse[1]){
        interpolate_stage<
            T1, T2, FP, decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse),  //
            false, false, true, LINEAR_BLOCK_SIZE, 8, 3, NO_COARSEN, 3, BORDER_INCLUSIVE, WORKFLOW>(
            s_data, s_ectrl,data_size, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[1]);
        interpolate_stage<
            T1, T2, FP, decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse),  //
            false, true, false, LINEAR_BLOCK_SIZE, 17, 2, NO_COARSEN, 3, BORDER_INCLUSIVE, WORKFLOW>(
            s_data, s_ectrl,data_size, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[1]);
        interpolate_stage<
            T1, T2, FP, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse),  //
            true, false, false, LINEAR_BLOCK_SIZE, 17, 5, NO_COARSEN, 2, BORDER_INCLUSIVE, WORKFLOW>(
            s_data, s_ectrl,data_size, xblue_reverse, yblue_reverse, zblue_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[1]);
    }
    else{
        interpolate_stage<
            T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
            true, false, false, LINEAR_BLOCK_SIZE, 9, 3, NO_COARSEN, 2, BORDER_INCLUSIVE, WORKFLOW>(
            s_data, s_ectrl,data_size, xblue, yblue, zblue, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[1]);
        interpolate_stage<
            T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
            false, true, false, LINEAR_BLOCK_SIZE, 9, 2, NO_COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
            s_data, s_ectrl,data_size, xyellow, yyellow, zyellow, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[1]);
        interpolate_stage<
            T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
            false, false, true, LINEAR_BLOCK_SIZE, 8, 5, NO_COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
            s_data, s_ectrl,data_size, xhollow, yhollow, zhollow, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[1]);

    }
    //if(TIX==0 and TIY==0 and TIZ==0 and BIX==0 and BIY==0 and BIZ==0)
    //printf("lv2\n");
    unit = 1;
    calc_eb(unit);
   // set_orders(reverse[0]);

    // iteration 3
    if(intp_param.reverse[0]){
        //may have bug 
        interpolate_stage<
            T1, T2, FP, decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse),  //
            false, false, true, LINEAR_BLOCK_SIZE, 16, 5, COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
            s_data, s_ectrl,data_size, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[0]);
        interpolate_stage<
            T1, T2, FP, decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse),  //
            false, true, false, LINEAR_BLOCK_SIZE, 33, 4, COARSEN, 5, BORDER_INCLUSIVE, WORKFLOW>(
            s_data, s_ectrl,data_size, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[0]);
        interpolate_stage<
            T1, T2, FP, decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse),  //
            true, false, false, LINEAR_BLOCK_SIZE, 33, 9, COARSEN, 4, BORDER_EXCLUSIVE, WORKFLOW>(
            s_data, s_ectrl,data_size, xblue_reverse, yblue_reverse, zblue_reverse, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[0]);

        //may have bug end
    }
    else{
        interpolate_stage<
            T1, T2, FP, decltype(xblue), decltype(yblue), decltype(zblue),  //
            true, false, false, LINEAR_BLOCK_SIZE, 17, 5, COARSEN, 4, BORDER_INCLUSIVE, WORKFLOW>(
            s_data, s_ectrl,data_size, xblue, yblue, zblue, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[0]);
        interpolate_stage<
            T1, T2, FP, decltype(xyellow), decltype(yyellow), decltype(zyellow),  //
            false, true, false, LINEAR_BLOCK_SIZE, 17, 4, COARSEN, 9, BORDER_INCLUSIVE, WORKFLOW>(
            s_data, s_ectrl,data_size, xyellow, yyellow, zyellow, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[0]);
       
        interpolate_stage<
            T1, T2, FP, decltype(xhollow), decltype(yhollow), decltype(zhollow),  //
            false, false, true, LINEAR_BLOCK_SIZE, 16, 9, COARSEN, 9, BORDER_EXCLUSIVE, WORKFLOW>(
            s_data, s_ectrl,data_size, xhollow, yhollow, zhollow, unit, cur_eb_r, cur_ebx2, radius, intp_param.interpolators[0]);

    }
  //  if(TIX==0 and TIY==0 and TIZ==0 and BIX==0 and BIY==0 and BIZ==0)
   // printf("lv1\n");
    


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

    /******************************************************************************
     test only: print a block
     ******************************************************************************/
    // if (TIX == 0 and BIX == 7 and BIY == 47 and BIZ == 15) { spline3d_print_block_from_GPU(s_ectrl); }
   //  if (TIX == 0 and BIX == 4 and BIY == 20 and BIZ == 20) { spline3d_print_block_from_GPU(s_data); }
}

/********************************************************************************
 * host API/kernel
 ********************************************************************************/

template <typename TITER, typename EITER, typename FP, int LINEAR_BLOCK_SIZE, typename CompactVal, typename CompactIdx, typename CompactNum>
__global__ void cusz::c_spline3d_infprecis_32x8x8data(
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
    INTERPOLATION_PARAMS intp_param)
{
    // compile time variables
    using T = typename std::remove_pointer<TITER>::type;
    using E = typename std::remove_pointer<EITER>::type;

    {
        __shared__ struct {
            T data[9][9][33];
            T ectrl[9][9][33];
        } shmem;

        c_reset_scratch_33x9x9data<T, T, LINEAR_BLOCK_SIZE>(shmem.data, shmem.ectrl, radius);
        //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
        //    printf("reset\n");
        //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
        //    printf("dsz: %d %d %d\n",data_size.x,data_size.y,data_size.z);

        global2shmem_33x9x9data<T, T, LINEAR_BLOCK_SIZE>(data, data_size, data_leap, shmem.data);

       // if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
        //    printf("g2s\n");
        // version 1, use shmem, erroneous
        // c_gather_anchor<T>(shmem.data, anchor, anchor_leap);
        // version 2, use global mem, correct
        c_gather_anchor<T>(data, data_size, data_leap, anchor, anchor_leap);

        cusz::device_api::spline3d_layout2_interpolate<T, T, FP,LINEAR_BLOCK_SIZE, SPLINE3_COMPR, false>(
            shmem.data, shmem.ectrl, data_size, eb_r, ebx2, radius, intp_param);
        //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
        //    printf("interp\n");
        //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
         //   printf("esz: %d %d %d\n",ectrl_size.x,ectrl_size.y,ectrl_size.z);


        //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)

        shmem2global_32x8x8data_with_compaction<T, E, LINEAR_BLOCK_SIZE>(shmem.ectrl, ectrl, ectrl_size, ectrl_leap, radius, compact_val, compact_idx, compact_num);

        // shmem2global_32x8x8data<T, E, LINEAR_BLOCK_SIZE>(shmem.ectrl, ectrl, ectrl_size, ectrl_leap);

        //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
        //    printf("s2g\n");
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
    int     radius,
    INTERPOLATION_PARAMS intp_param)
{
    // compile time variables
    using E = typename std::remove_pointer<EITER>::type;
    using T = typename std::remove_pointer<TITER>::type;

    __shared__ struct {
        T data[9][9][33];
        T ectrl[9][9][33];
    } shmem;

    x_reset_scratch_33x9x9data<T, T, LINEAR_BLOCK_SIZE>(shmem.data, shmem.ectrl, anchor, anchor_size, anchor_leap);

    //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
    //        printf("esz: %d %d %d\n",ectrl_size.x,ectrl_size.y,ectrl_size.z);

    // global2shmem_33x9x9data<E, T, LINEAR_BLOCK_SIZE>(ectrl, ectrl_size, ectrl_leap, shmem.ectrl);
    global2shmem_fuse<T, E, LINEAR_BLOCK_SIZE>(ectrl, ectrl_size, ectrl_leap, data, shmem.ectrl);

    cusz::device_api::spline3d_layout2_interpolate<T, T, FP, LINEAR_BLOCK_SIZE, SPLINE3_DECOMPR, false>(
        shmem.data, shmem.ectrl, data_size, eb_r, ebx2, radius, intp_param);
    //if(TIX==0 and BIX==0 and BIY==0 and BIZ==0)
    //        printf("dsz: %d %d %d\n",data_size.x,data_size.y,data_size.z);
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
#undef GDX
#undef GDY
#undef GDZ

#endif
