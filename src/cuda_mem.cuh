// 20-04-30

#ifndef CUDA_MEM_CUH
#define CUDA_MEM_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace mem {
template <typename T>
T* CreateCUDASpace(size_t l, uint8_t i = 0);

template <typename T>
T* CreateDeviceSpaceAndMemcpyFromHost(T* var, size_t l);

template <typename T>
T* CreateHostSpaceAndMemcpyFromDevice(T* d_var, size_t l);
}  // namespace mem

#endif
