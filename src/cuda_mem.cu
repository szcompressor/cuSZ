// 20-04-30

#include <cuda_runtime.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include "cuda_mem.cuh"

template <typename T>
inline T* mem::CreateCUDASpace(size_t l, uint8_t i)
{
    T* d_var;
    cudaMalloc(&d_var, l * sizeof(T));
    cudaMemset(d_var, i, l * sizeof(T));
    return d_var;
}

// enum MemcpyDirection { h2d, d2h };

template <typename T>
void mem::CopyBetweenSpaces(T* src, T* dst, size_t l, MemcpyDirection direct)
{
    assert(src != nullptr);
    assert(dst != nullptr);
    if (direct == h2d) {
        cudaMemcpy(dst, src, sizeof(T) * l, cudaMemcpyHostToDevice);
    }
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

template uint8_t*  mem::CreateCUDASpace<uint8_t>(size_t l, uint8_t i);
template uint16_t* mem::CreateCUDASpace<uint16_t>(size_t l, uint8_t i);
template uint32_t* mem::CreateCUDASpace<uint32_t>(size_t l, uint8_t i);
template uint64_t* mem::CreateCUDASpace<uint64_t>(size_t l, uint8_t i);
template int8_t*   mem::CreateCUDASpace<int8_t>(size_t l, uint8_t i);
template int16_t*  mem::CreateCUDASpace<int16_t>(size_t l, uint8_t i);
template int32_t*  mem::CreateCUDASpace<int32_t>(size_t l, uint8_t i);
template int64_t*  mem::CreateCUDASpace<int64_t>(size_t l, uint8_t i);
template float*    mem::CreateCUDASpace<float>(size_t l, uint8_t i);
template double*   mem::CreateCUDASpace<double>(size_t l, uint8_t i);

template int8_t*   mem::CreateDeviceSpaceAndMemcpyFromHost(int8_t* var, size_t l);
template int16_t*  mem::CreateDeviceSpaceAndMemcpyFromHost(int16_t* var, size_t l);
template int32_t*  mem::CreateDeviceSpaceAndMemcpyFromHost(int32_t* var, size_t l);
template int64_t*  mem::CreateDeviceSpaceAndMemcpyFromHost(int64_t* var, size_t l);
template uint8_t*  mem::CreateDeviceSpaceAndMemcpyFromHost(uint8_t* var, size_t l);
template uint16_t* mem::CreateDeviceSpaceAndMemcpyFromHost(uint16_t* var, size_t l);
template uint32_t* mem::CreateDeviceSpaceAndMemcpyFromHost(uint32_t* var, size_t l);
template uint64_t* mem::CreateDeviceSpaceAndMemcpyFromHost(uint64_t* var, size_t l);
template float*    mem::CreateDeviceSpaceAndMemcpyFromHost(float* var, size_t l);
template double*   mem::CreateDeviceSpaceAndMemcpyFromHost(double* var, size_t l);

template int8_t*   mem::CreateHostSpaceAndMemcpyFromDevice(int8_t* d_var, size_t l);
template int16_t*  mem::CreateHostSpaceAndMemcpyFromDevice(int16_t* d_var, size_t l);
template int32_t*  mem::CreateHostSpaceAndMemcpyFromDevice(int32_t* d_var, size_t l);
template int64_t*  mem::CreateHostSpaceAndMemcpyFromDevice(int64_t* d_var, size_t l);
template uint8_t*  mem::CreateHostSpaceAndMemcpyFromDevice(uint8_t* d_var, size_t l);
template uint16_t* mem::CreateHostSpaceAndMemcpyFromDevice(uint16_t* d_var, size_t l);
template uint32_t* mem::CreateHostSpaceAndMemcpyFromDevice(uint32_t* d_var, size_t l);
template uint64_t* mem::CreateHostSpaceAndMemcpyFromDevice(uint64_t* d_var, size_t l);
template float*    mem::CreateHostSpaceAndMemcpyFromDevice(float* d_var, size_t l);
template double*   mem::CreateHostSpaceAndMemcpyFromDevice(double* d_var, size_t l);
