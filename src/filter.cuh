#ifndef FILTER_CUH
#define FILTER_CUH

#include <iostream>
#include "stdio.h"
using std::cout;
using std::endl;

namespace Prototype {

template <typename T>
__global__ void binning2d_2x2_eveneven(T* input, T* output, size_t dim0, size_t dim1, size_t new_dim0, size_t new_dim1) {
    size_t yid = blockIdx.y * blockDim.y + threadIdx.y;
    size_t xid = blockIdx.x * blockDim.x + threadIdx.x;

    if (yid >= new_dim1 or xid >= new_dim0) return;

    auto NW = *(input + (2 * yid) * dim0 + (2 * xid));
    auto NE = *(input + (2 * yid) * dim0 + (2 * xid + 1));
    auto SW = *(input + (2 * yid + 1) * dim0 + (2 * xid));
    auto SE = *(input + (2 * yid + 1) * dim0 + (2 * xid + 1));

    *(output + yid * new_dim0 + xid) = (NW + NE + SW + SE) / 4;
}

template <typename T, int DS, int tBLK>
__global__ void binning2d(T* input, T* output, size_t d0, size_t d1, size_t new_d0, size_t new_d1) {
    auto         y   = threadIdx.y;
    auto         x   = threadIdx.x;
    auto         yid = blockIdx.y * blockDim.y + y;
    auto         xid = blockIdx.x * blockDim.x + x;
    __shared__ T s[tBLK][tBLK];

    if (yid >= new_d1 or xid >= new_d0) return;

    int xblk = (xid + 1) * DS >= d0 ? d0 - xid * DS : DS;
    int yblk = (yid + 1) * DS >= d1 ? d1 - yid * DS : DS;
    s[y][x]  = 0;

    for (int j = 0; j < yblk; j++)
        for (int i = 0; i < xblk; i++) s[y][x] += input[(yid * DS + j) * d0 + (xid * DS + i)];

    output[yid * new_d0 + xid] = s[y][x] / static_cast<T>(yblk * xblk);
}

template <typename T, int BLK>
__host__ void binning2d(T* input, T* output, size_t dim0, size_t dim1, size_t new_dim0, size_t new_dim1, size_t xid, size_t yid) {
    if (yid >= new_dim1 or xid >= new_dim0) return;

    int xblk = (xid + 1) * BLK >= dim0 ? dim0 - xid * BLK : BLK;
    int yblk = (yid + 1) * BLK >= dim1 ? dim1 - yid * BLK : BLK;

    T sum = 0;
    for (int j = 0; j < yblk; j++)
        for (int i = 0; i < xblk; i++) sum += input[(yid * BLK + j) * dim0 + (xid * BLK + i)];

    output[yid * new_dim0 + xid] = sum / static_cast<T>(yblk * xblk);
}

}  // namespace Prototype

#endif
