/**
 * @file cusz_dryrun.cu
 * @author Jiannan Tian
 * @brief cuSZ dryrun mode, checking data quality from lossy compression.
 * @version 0.2
 * @date 2020-09-20
 * (create) 2020-05-14, (release) 2020-09-20, (rev1) 2021-01-25
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include "dryrun.cuh"
#include "metadata.hh"

#define tix threadIdx.x
#define tiy threadIdx.y
#define tiz threadIdx.z
#define bix blockIdx.x
#define biy blockIdx.y
#define biz blockIdx.z
#define bdx blockDim.x
#define bdy blockDim.y
#define bdz blockDim.z

template <typename Data>
__global__ void cusz::dryrun::lorenzo_1d1l(lorenzo_dryrun ctx, Data* d)
{
    auto id = bix * bdx + tix;
    if (id >= ctx.d0) return;
    d[id] = round(d[id] * ctx.ebx2_r) * ctx.ebx2;  // prequant
}

template <typename Data>
__global__ void cusz::dryrun::lorenzo_2d1l(lorenzo_dryrun ctx, Data* d)
{
    auto   gi1 = biy * bdy + tiy;
    auto   gi0 = bix * bdx + tix;
    size_t id  = gi0 + gi1 * ctx.stride1;  // low to high dim, inner to outer
    if (gi0 >= ctx.d0 or gi1 >= ctx.d1) return;
    d[id] = round(d[id] * ctx.ebx2_r) * ctx.ebx2;  // prequant
}

template <typename Data>
__global__ void cusz::dryrun::lorenzo_3d1l(lorenzo_dryrun ctx, Data* d)
{
    auto   gi2 = biz * bdz + tiz;
    auto   gi1 = biy * bdy + tiy;
    auto   gi0 = bix * bdx + tix;
    size_t id  = gi0 + gi1 * ctx.stride1 + gi2 * ctx.stride2;  // low to high in dim, inner to outer
    if (gi0 >= ctx.d0 or gi1 >= ctx.d1 or gi2 >= ctx.d2) return;
    d[id] = round(d[id] * ctx.ebx2_r) * ctx.ebx2;  // prequant
}

template __global__ void cusz::dryrun::lorenzo_1d1l<float>(lorenzo_dryrun, float*);
template __global__ void cusz::dryrun::lorenzo_2d1l<float>(lorenzo_dryrun, float*);
template __global__ void cusz::dryrun::lorenzo_3d1l<float>(lorenzo_dryrun, float*);
