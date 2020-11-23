/**
 * @file filter.cu
 * @author Jiannan Tian
 * @brief Filters for preprocessing of cuSZ.
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-05-03
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <iostream>
#include "filter.cuh"
#include "stdio.h"
#include "ad_hoc_types.hh"

using std::cout;
using std::endl;

template <typename Data, int DownscaleFactor, int tBLK>
__global__ void Prototype::binning2d(Data* input, Data* output, size_t d0, size_t d1, size_t new_d0, size_t new_d1)
{
    auto y   = threadIdx.y;
    auto x   = threadIdx.x;
    auto yid = blockIdx.y * blockDim.y + y;
    auto xid = blockIdx.x * blockDim.x + x;

    __shared__ Data s[tBLK][tBLK];

    if (yid >= new_d1 or xid >= new_d0) return;

    int xblk = (xid + 1) * DownscaleFactor >= d0 ? d0 - xid * DownscaleFactor : DownscaleFactor;
    int yblk = (yid + 1) * DownscaleFactor >= d1 ? d1 - yid * DownscaleFactor : DownscaleFactor;
    s[y][x]  = 0;

    for (int j = 0; j < yblk; j++)
        for (int i = 0; i < xblk; i++) s[y][x] += input[(yid * DownscaleFactor + j) * d0 + (xid * DownscaleFactor + i)];

    output[yid * new_d0 + xid] = s[y][x] / static_cast<Data>(yblk * xblk);
}

template __global__ void Prototype::binning2d<FP4, 2, 32>(FP4*, FP4*, size_t, size_t, size_t, size_t);
template __global__ void Prototype::binning2d<FP8, 2, 32>(FP8*, FP8*, size_t, size_t, size_t, size_t);
template __global__ void Prototype::binning2d<I1, 2, 32>(I1*, I1*, size_t, size_t, size_t, size_t);
template __global__ void Prototype::binning2d<I2, 2, 32>(I2*, I2*, size_t, size_t, size_t, size_t);
template __global__ void Prototype::binning2d<I4, 2, 32>(I4*, I4*, size_t, size_t, size_t, size_t);
template __global__ void Prototype::binning2d<I8, 2, 32>(I8*, I8*, size_t, size_t, size_t, size_t);
template __global__ void Prototype::binning2d<I8_2, 2, 32>(I8_2*, I8_2*, size_t, size_t, size_t, size_t);