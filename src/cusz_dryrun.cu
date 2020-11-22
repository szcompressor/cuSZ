/**
 * @file cusz_dryrun.cu
 * @author Jiannan Tian
 * @brief cuSZ dryrun mode, checking data quality from lossy compression.
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-05-14
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <iostream>
#include <string>
#include "cuda_mem.cuh"
#include "cusz_dryrun.cuh"
#include "cusz_dualquant.cuh"
#include "format.hh"
#include "io.hh"
#include "verify.hh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

const int DIM0 = 0;
const int DIM1 = 1;
const int DIM2 = 2;
// const int DIM3   = 3;
const int nBLK0 = 4;
const int nBLK1 = 5;
const int nBLK2 = 6;
// const int nBLK3  = 7;
const int nDIM = 8;
const int LEN  = 12;
// const int CAP    = 13;
// const int RADIUS = 14;

const size_t EB = 0;
// const size_t EBr    = 1;
const size_t EBx2   = 2;
const size_t EBx2_r = 3;

const int B_1d = 32;
const int B_2d = 16;
const int B_3d = 8;

template <typename Data>
__global__ void cusz::dryrun::lorenzo_1d1l(Data* d, const size_t* dims, const double* eb_variants)
{
    auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= dims[DIM0]) return;
    d[id] = round(d[id] * eb_variants[EBx2_r]) * eb_variants[EBx2];  // prequant
}

template <typename Data>
__global__ void cusz::dryrun::lorenzo_2d1l(Data* d, const size_t* dims, const double* eb_variants)
{
    auto   y   = threadIdx.y;
    auto   x   = threadIdx.x;
    auto   gi1 = blockIdx.y * blockDim.y + y;
    auto   gi0 = blockIdx.x * blockDim.x + x;
    size_t id  = gi0 + gi1 * dims[DIM0];  // low to high dim, inner to outer
    if (gi0 >= dims[DIM0] or gi1 >= dims[DIM1]) return;
    d[id] = round(d[id] * eb_variants[EBx2_r]) * eb_variants[EBx2];  // prequant
}

template <typename Data>
__global__ void cusz::dryrun::lorenzo_3d1l(Data* d, const size_t* dims, const double* eb_variants)
{
    auto   gi2 = blockIdx.z * blockDim.z + threadIdx.z;
    auto   gi1 = blockIdx.y * blockDim.y + threadIdx.y;
    auto   gi0 = blockIdx.x * blockDim.x + threadIdx.x;
    size_t id  = gi0 + gi1 * dims[DIM0] + gi2 * dims[DIM0] * dims[DIM1];  // low to high in dim, inner to outer
    if (gi0 >= dims[DIM0] or gi1 >= dims[DIM1] or gi2 >= dims[DIM2]) return;
    d[id] = round(d[id] * eb_variants[EBx2_r]) * eb_variants[EBx2];  // prequant
}

template <typename Data>
void cusz::interface::DryRun(Data* d, Data* d_d, const string& fi, size_t* dims, double* ebs)
{
    auto len           = dims[LEN];
    auto d_dims        = mem::CreateDeviceSpaceAndMemcpyFromHost(dims, 16);
    auto d_eb_variants = mem::CreateDeviceSpaceAndMemcpyFromHost(ebs, 4);

    auto d2 = new Data[len]();
    memcpy(d2, d, sizeof(Data) * len);

    if (dims[nDIM] == 1) {
        dim3 block_num(dims[nBLK0]);
        dim3 thread_num(B_1d);
        cusz::dryrun::lorenzo_1d1l<Data><<<block_num, thread_num>>>(d_d, d_dims, d_eb_variants);
    }
    else if (dims[nDIM] == 2) {
        dim3 block_num(dims[nBLK0], dims[nBLK1]);
        dim3 thread_num(B_2d, B_2d);
        cusz::dryrun::lorenzo_2d1l<Data><<<block_num, thread_num>>>(d_d, d_dims, d_eb_variants);
    }
    else if (dims[nDIM] == 3) {
        dim3 block_num(dims[nBLK0], dims[nBLK1], dims[nBLK2]);
        dim3 thread_num(B_3d, B_3d, B_3d);
        cusz::dryrun::lorenzo_3d1l<Data><<<block_num, thread_num>>>(d_d, d_dims, d_eb_variants);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(d, d_d, len * sizeof(Data), cudaMemcpyDeviceToHost);

    analysis::VerifyData<Data>(d, d2, len, false, ebs[EB], 0);
    delete[] d2;
    cudaFree(d_d);
    cudaFree(d_dims);
    cudaFree(d_eb_variants);
}

template __global__ void cusz::dryrun::lorenzo_1d1l<float>(float*, const size_t*, const double*);
template __global__ void cusz::dryrun::lorenzo_2d1l<float>(float*, const size_t*, const double*);
template __global__ void cusz::dryrun::lorenzo_3d1l<float>(float*, const size_t*, const double*);

template void cusz::interface::DryRun<float>(float* d, float* d_d, const string& fi, size_t* dims, double* ebs);
