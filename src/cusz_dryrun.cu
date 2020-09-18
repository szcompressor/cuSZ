// jtian 20-05-14

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

template <typename T>
__global__ void cusz::dryrun::lorenzo_1d1l(T* data, size_t* dims_L16, double* ebs_L4)
{
    auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= dims_L16[DIM0]) return;
    data[id] = round(data[id] * ebs_L4[EBx2_r]) * ebs_L4[EBx2];  // prequantization
}

template <typename T>
__global__ void cusz::dryrun::lorenzo_2d1l(T* data, size_t* dims_L16, double* ebs_L4)
{
    auto   y   = threadIdx.y;
    auto   x   = threadIdx.x;
    auto   gi1 = blockIdx.y * blockDim.y + y;
    auto   gi0 = blockIdx.x * blockDim.x + x;
    size_t id  = gi0 + gi1 * dims_L16[DIM0];  // low to high dim, inner to outer
    if (gi0 >= dims_L16[DIM0] or gi1 >= dims_L16[DIM1]) return;
    data[id] = round(data[id] * ebs_L4[EBx2_r]) * ebs_L4[EBx2];  // prequantization
}

template <typename T>
__global__ void cusz::dryrun::lorenzo_3d1l(T* data, size_t* dims_L16, double* ebs_L4)
{
    auto   gi2 = blockIdx.z * blockDim.z + threadIdx.z;
    auto   gi1 = blockIdx.y * blockDim.y + threadIdx.y;
    auto   gi0 = blockIdx.x * blockDim.x + threadIdx.x;
    size_t id  = gi0 + gi1 * dims_L16[DIM0] + gi2 * dims_L16[DIM0] * dims_L16[DIM1];  // low to high in dim, inner to outer
    if (gi0 >= dims_L16[DIM0] or gi1 >= dims_L16[DIM1] or gi2 >= dims_L16[DIM2]) return;
    data[id] = round(data[id] * ebs_L4[EBx2_r]) * ebs_L4[EBx2];  // prequantization
}

template <typename T>
void cusz::workflow::DryRun(T* d, T* d_d, string fi, size_t* dims, double* ebs)
{
    cout << log_info << "Entering dry-run mode..." << endl;
    auto len        = dims[LEN];
    auto d_dims_L16 = mem::CreateDeviceSpaceAndMemcpyFromHost(dims, 16);
    auto d_ebs_L4   = mem::CreateDeviceSpaceAndMemcpyFromHost(ebs, 4);

    if (dims[nDIM] == 1) {
        dim3 blockNum(dims[nBLK0]);
        dim3 threadNum(B_1d);
        cusz::dryrun::lorenzo_1d1l<T><<<blockNum, threadNum>>>(d_d, d_dims_L16, d_ebs_L4);
    }
    else if (dims[nDIM] == 2) {
        dim3 blockNum(dims[nBLK0], dims[nBLK1]);
        dim3 threadNum(B_2d, B_2d);
        cusz::dryrun::lorenzo_2d1l<T><<<blockNum, threadNum>>>(d_d, d_dims_L16, d_ebs_L4);
    }
    else if (dims[nDIM] == 3) {
        dim3 blockNum(dims[nBLK0], dims[nBLK1], dims[nBLK2]);
        dim3 threadNum(B_3d, B_3d, B_3d);
        cusz::dryrun::lorenzo_3d1l<T><<<blockNum, threadNum>>>(d_d, d_dims_L16, d_ebs_L4);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(d, d_d, len * sizeof(T), cudaMemcpyDeviceToHost);

    auto d2 = io::ReadBinaryFile<T>(fi, len);
    // CR is not valid in dry run
    analysis::VerifyData<T>(d, d2, len, false, ebs[EB], 0);
    cout << log_info << "Dry-run finished, exit..." << endl;
    delete[] d;
    delete[] d2;
    cudaFree(d_d);
    cudaFree(d_dims_L16);
    cudaFree(d_ebs_L4);
}

template __global__ void cusz::dryrun::lorenzo_1d1l<float>(float*, size_t*, double*);
template __global__ void cusz::dryrun::lorenzo_2d1l<float>(float*, size_t*, double*);
template __global__ void cusz::dryrun::lorenzo_3d1l<float>(float*, size_t*, double*);

template void cusz::workflow::DryRun<float>(float* d, float* d_d, string fi, size_t* dims, double* ebs);
/*
template void cusz::workflow::DryRun<double>(double* d, double* d_d, string fi, size_t* dims, double* ebs);
template void cusz::workflow::DryRun<char>(char* d, char* d_d, string fi, size_t* dims, double* ebs);
template void cusz::workflow::DryRun<short>(short* d, short* d_d, string fi, size_t* dims, double* ebs);
template void cusz::workflow::DryRun<int>(int* d, int* d_d, string fi, size_t* dims, double* ebs);
template void cusz::workflow::DryRun<long>(long* d, long* d_d, string fi, size_t* dims, double* ebs);
template void cusz::workflow::DryRun<long long>(long long* d, long long* d_d, string fi, size_t* dims, double* ebs);
 */
