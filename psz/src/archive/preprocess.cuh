/**
 * @file preprocess.cuh
 * @author Jiannan Tian
 * @brief Filters for preprocessing of cuSZ.
 * @version 0.3
 * @date 2020-09-20
 * (created) 2020-05-03 (rev) 2021-06-21
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_KERNEL_PREPROCESS_CUH
#define CUSZ_KERNEL_PREPROCESS_CUH

#include <iostream>

#include "typing.hh"
#include "utils/configs.hh"

using std::cout;
using std::endl;

namespace cusz {

#include <numeric>

template <typename T>
__global__ void log_transform()
{
    static_assert(std::is_floating_point<T>::value, "[log_transform] must be floating-point type.");
}

template <typename Data, int DOWNSCALE_FACTOR, int tBLK>
__global__ void binning2d(Data* input, Data* output, size_t d0, size_t d1, size_t new_d0, size_t new_d1)
{
    auto y   = threadIdx.y;
    auto x   = threadIdx.x;
    auto yid = blockIdx.y * blockDim.y + y;
    auto xid = blockIdx.x * blockDim.x + x;

    __shared__ Data s[tBLK][tBLK];

    if (yid >= new_d1 or xid >= new_d0) return;

    int xblk = (xid + 1) * DOWNSCALE_FACTOR >= d0 ? d0 - xid * DOWNSCALE_FACTOR : DOWNSCALE_FACTOR;
    int yblk = (yid + 1) * DOWNSCALE_FACTOR >= d1 ? d1 - yid * DOWNSCALE_FACTOR : DOWNSCALE_FACTOR;
    s[y][x]  = 0;

    for (int j = 0; j < yblk; j++)
        for (int i = 0; i < xblk; i++)
            s[y][x] += input[(yid * DOWNSCALE_FACTOR + j) * d0 + (xid * DOWNSCALE_FACTOR + i)];

    output[yid * new_d0 + xid] = s[y][x] / static_cast<Data>(yblk * xblk);
}
}  // namespace cusz

template __global__ void cusz::binning2d<float, 2, 32>(float*, float*, size_t, size_t, size_t, size_t);
template __global__ void cusz::binning2d<double, 2, 32>(double*, double*, size_t, size_t, size_t, size_t);
// template __global__ void cusz::binning2d<I1, 2, 32>(I1*, I1*, size_t, size_t, size_t, size_t);
// template __global__ void cusz::binning2d<I2, 2, 32>(I2*, I2*, size_t, size_t, size_t, size_t);
// template __global__ void cusz::binning2d<I4, 2, 32>(I4*, I4*, size_t, size_t, size_t, size_t);
// template __global__ void cusz::binning2d<I8, 2, 32>(I8*, I8*, size_t, size_t, size_t, size_t);

#endif
