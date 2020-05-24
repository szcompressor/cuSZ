// jtian 20-05-03

#include <iostream>
#include "filter.cuh"
#include "stdio.h"

using std::cout;
using std::endl;

template <typename T, int DS, int tBLK>
__global__ void Prototype::binning2d(T* input, T* output, size_t d0, size_t d1, size_t new_d0, size_t new_d1)
{
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

template __global__ void Prototype::binning2d<float, 2, 32>(float* input, float* output, size_t d0, size_t d1, size_t new_d0, size_t new_d1);
template __global__ void Prototype::binning2d<double, 2, 32>(double* input, double* output, size_t d0, size_t d1, size_t new_d0, size_t new_d1);
template __global__ void Prototype::binning2d<char, 2, 32>(char* input, char* output, size_t d0, size_t d1, size_t new_d0, size_t new_d1);
template __global__ void Prototype::binning2d<short, 2, 32>(short* input, short* output, size_t d0, size_t d1, size_t new_d0, size_t new_d1);
template __global__ void Prototype::binning2d<int, 2, 32>(int* input, int* output, size_t d0, size_t d1, size_t new_d0, size_t new_d1);
template __global__ void Prototype::binning2d<long, 2, 32>(long* input, long* output, size_t d0, size_t d1, size_t new_d0, size_t new_d1);
template __global__ void Prototype::binning2d<long long, 2, 32>(long long* input, long long* output, size_t d0, size_t d1, size_t new_d0, size_t new_d1);

/*
template <typename T, int BLK>
__host__ void Prototype::binning2d(T* input, T* output, size_t dim0, size_t dim1, size_t new_dim0, size_t new_dim1, size_t xid, size_t yid) {
    if (yid >= new_dim1 or xid >= new_dim0) return;

    int xblk = (xid + 1) * BLK >= dim0 ? dim0 - xid * BLK : BLK;
    int yblk = (yid + 1) * BLK >= dim1 ? dim1 - yid * BLK : BLK;

    T sum = 0;
    for (int j = 0; j < yblk; j++)
        for (int i = 0; i < xblk; i++) sum += input[(yid * BLK + j) * dim0 + (xid * BLK + i)];

    output[yid * new_dim0 + xid] = sum / static_cast<T>(yblk * xblk);
}
 */
