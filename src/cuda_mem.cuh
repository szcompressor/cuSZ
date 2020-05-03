// 20-04-30

#ifndef CUDA_MEM_CUH
#define CUDA_MEM_CUH

#include <cuda_runtime.h>

namespace mem {
template <typename T>
T* CreateCUDASpace(size_t l, uint8_t i = 0) {
    T* d_var;
    cudaMalloc(&d_var, l * sizeof(T));
    cudaMemset(d_var, i, l * sizeof(T));
    return d_var;
}
template <typename T>
T* CreateDeviceSpaceAndMemcpyFromHost(T* var, size_t l) {
    T* d_var;
    cudaMalloc(&d_var, l * sizeof(T));
    cudaMemcpy(d_var, var, l * sizeof(T), cudaMemcpyHostToDevice);
    return d_var;
}
template <typename T>
T* CreateHostSpaceAndMemcpyFromDevice(T* d_var, size_t l) {
    auto var = new T[l];
    cudaMemcpy(var, d_var, l * sizeof(T), cudaMemcpyDeviceToHost);
    return var;
}
}  // namespace mem

#endif
