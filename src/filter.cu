/**
 * @file filter.cu
 * @author Jiannan Tian
 * @brief Filters for preprocessing of cuSZ.
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-05-03
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <iostream>
#include "filter.cuh"
#include "stdio.h"

using std::cout;
using std::endl;

template <typename T, int DS, int tBLK>
__global__ void Prototype::binning2d(T* input, T* output, size_t d0, size_t d1, size_t new_d0, size_t new_d1)
{
    auto y   = threadIdx.y;
    auto x   = threadIdx.x;
    auto yid = blockIdx.y * blockDim.y + y;
    auto xid = blockIdx.x * blockDim.x + x;

    __shared__ T s[tBLK][tBLK];

    if (yid >= new_d1 or xid >= new_d0) return;

    int xblk = (xid + 1) * DS >= d0 ? d0 - xid * DS : DS;
    int yblk = (yid + 1) * DS >= d1 ? d1 - yid * DS : DS;
    s[y][x]  = 0;

    for (int j = 0; j < yblk; j++)
        for (int i = 0; i < xblk; i++) s[y][x] += input[(yid * DS + j) * d0 + (xid * DS + i)];

    output[yid * new_d0 + xid] = s[y][x] / static_cast<T>(yblk * xblk);
}

template __global__ void Prototype::binning2d<float, 2, 32>(float*, float*, size_t, size_t, size_t, size_t);
template __global__ void Prototype::binning2d<double, 2, 32>(double*, double*, size_t, size_t, size_t, size_t);
template __global__ void Prototype::binning2d<char, 2, 32>(char*, char*, size_t, size_t, size_t, size_t);
template __global__ void Prototype::binning2d<short, 2, 32>(short*, short*, size_t, size_t, size_t, size_t);
template __global__ void Prototype::binning2d<int, 2, 32>(int*, int*, size_t, size_t, size_t, size_t);
template __global__ void Prototype::binning2d<long, 2, 32>(long*, long*, size_t, size_t, size_t, size_t);
template __global__ void Prototype::binning2d<long long, 2, 32>(long long*, long long*, size_t, size_t, size_t, size_t);
