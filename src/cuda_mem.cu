/**
 * @file cuda_mem.cu
 * @author Jiannan Tian
 * @brief CUDA memory operation wrappers.
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-04-30
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cuda_runtime.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include "cuda_mem.cuh"
#include "type_aliasing.hh"

template <typename T>
inline T* mem::CreateCUDASpace(size_t len, uint8_t filling_val)
{
    T* d_var;
    cudaMalloc(&d_var, len * sizeof(T));
    cudaMemset(d_var, filling_val, len * sizeof(T));
    return d_var;
}

// enum MemcpyDirection { h2d, d2h };

template <typename T>
void mem::CopyBetweenSpaces(T* src, T* dst, size_t l, MemcpyDirection direct)
{
    assert(src != nullptr);
    assert(dst != nullptr);
    if (direct == h2d) { cudaMemcpy(dst, src, sizeof(T) * l, cudaMemcpyHostToDevice); }
    else if (direct == d2h) {
        cudaMemcpy(dst, src, sizeof(T) * l, cudaMemcpyDeviceToHost);
    }
    else {
        // TODO log
        exit(1);
    }
}

template <typename T>
inline T* mem::CreateDeviceSpaceAndMemcpyFromHost(T* var, size_t l)
{
    T* d_var;
    cudaMalloc(&d_var, l * sizeof(T));
    cudaMemcpy(d_var, var, l * sizeof(T), cudaMemcpyHostToDevice);
    return d_var;
}
template <typename T>
inline T* mem::CreateHostSpaceAndMemcpyFromDevice(T* d_var, size_t l)
{
    auto var = new T[l];
    cudaMemcpy(var, d_var, l * sizeof(T), cudaMemcpyDeviceToHost);
    return var;
}

template I1*  mem::CreateCUDASpace<I1>(size_t len, UI1 filling_val);
template I2*  mem::CreateCUDASpace<I2>(size_t len, UI1 filling_val);
template I4*  mem::CreateCUDASpace<I4>(size_t len, UI1 filling_val);
template I8*  mem::CreateCUDASpace<I8>(size_t len, UI1 filling_val);
template UI1* mem::CreateCUDASpace<UI1>(size_t len, UI1 filling_val);
template UI2* mem::CreateCUDASpace<UI2>(size_t len, UI1 filling_val);
template UI4* mem::CreateCUDASpace<UI4>(size_t len, UI1 filling_val);
template UI8* mem::CreateCUDASpace<UI8>(size_t len, UI1 filling_val);
template FP4* mem::CreateCUDASpace<FP4>(size_t len, UI1 filling_val);
template FP8* mem::CreateCUDASpace<FP8>(size_t len, UI1 filling_val);
//
template I8_2*  mem::CreateCUDASpace<I8_2>(size_t len, UI1 filling_val);
template UI8_2* mem::CreateCUDASpace<UI8_2>(size_t len, UI1 filling_val);

template I1*  mem::CreateDeviceSpaceAndMemcpyFromHost(I1* var, size_t l);
template I2*  mem::CreateDeviceSpaceAndMemcpyFromHost(I2* var, size_t l);
template I4*  mem::CreateDeviceSpaceAndMemcpyFromHost(I4* var, size_t l);
template I8*  mem::CreateDeviceSpaceAndMemcpyFromHost(I8* var, size_t l);
template UI1* mem::CreateDeviceSpaceAndMemcpyFromHost(UI1* var, size_t l);
template UI2* mem::CreateDeviceSpaceAndMemcpyFromHost(UI2* var, size_t l);
template UI4* mem::CreateDeviceSpaceAndMemcpyFromHost(UI4* var, size_t l);
template UI8* mem::CreateDeviceSpaceAndMemcpyFromHost(UI8* var, size_t l);
template FP4* mem::CreateDeviceSpaceAndMemcpyFromHost(FP4* var, size_t l);
template FP8* mem::CreateDeviceSpaceAndMemcpyFromHost(FP8* var, size_t l);
//
template UI8_2* mem::CreateDeviceSpaceAndMemcpyFromHost(UI8_2* var, size_t l);
template I8_2*  mem::CreateDeviceSpaceAndMemcpyFromHost(I8_2* var, size_t l);

template I1*  mem::CreateHostSpaceAndMemcpyFromDevice(I1* d_var, size_t l);
template I2*  mem::CreateHostSpaceAndMemcpyFromDevice(I2* d_var, size_t l);
template I4*  mem::CreateHostSpaceAndMemcpyFromDevice(I4* d_var, size_t l);
template I8*  mem::CreateHostSpaceAndMemcpyFromDevice(I8* d_var, size_t l);
template UI1* mem::CreateHostSpaceAndMemcpyFromDevice(UI1* d_var, size_t l);
template UI2* mem::CreateHostSpaceAndMemcpyFromDevice(UI2* d_var, size_t l);
template UI4* mem::CreateHostSpaceAndMemcpyFromDevice(UI4* d_var, size_t l);
template UI8* mem::CreateHostSpaceAndMemcpyFromDevice(UI8* d_var, size_t l);
template FP4* mem::CreateHostSpaceAndMemcpyFromDevice(FP4* d_var, size_t l);
template FP8* mem::CreateHostSpaceAndMemcpyFromDevice(FP8* d_var, size_t l);
//
template I8_2*  mem::CreateHostSpaceAndMemcpyFromDevice(I8_2* d_var, size_t l);
template UI8_2* mem::CreateHostSpaceAndMemcpyFromDevice(UI8_2* d_var, size_t l);
