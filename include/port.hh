#if defined(PSZ_USE_CUDA)

// #include <cuda_runtime.h>
#define GpuStreamT cudaStream_t
#define GpuStreamCreate cudaStreamCreate
#define GpuStreamDestroy cudaStreamDestroy

#define GpuDeviceSync cudaDeviceSynchronize
#define GpuStreamSync(STREAM) cudaStreamSynchronize((cudaStream_t)STREAM)

#define GpuMemcpy cudaMemcpy
#define GpuMemcpyAsync cudaMemcpyAsync
#define GpuMemcpyKind cudaMemcpyKind
#define GpuMemcpyH2D cudaMemcpyHostToDevice
#define GpuMemcpyH2H cudaMemcpyHostToHost
#define GpuMemcpyD2H cudaMemcpyDeviceToHost
#define GpuMemcpyD2D cudaMemcpyDeviceToDevice

#define GpuMalloc cudaMalloc
#define GpuFree cudaFree
#define GpuFreeHost cudaFreeHost

#define GpuGetDevice cudaGetDevice
#define GpuSetDevice cudaSetDevice
#define GpuDeviceProp cudaDeviceProp
#define GpuGetDeviceProperties cudaGetDeviceProperties
#define GpuDeviceGetAttribute cudaDeviceGetAttribute
#define GpuDevicePtr CUdeviceptr
#define GpuMemGetAddressRange cuMemGetAddressRange
#define GpuFuncSetAttribute cudaFuncSetAttribute
#define GpuFuncAttribute cudaFuncAttribute

#define GpuDevAttrMultiProcessorCount cudaDevAttrMultiProcessorCount
#define GpuDevAttrMaxSharedMemoryPerBlock cudaDevAttrMaxSharedMemoryPerBlock
#define GpuDevAttrMaxSharedMemoryPerBlockOptin cudaDevAttrMaxSharedMemoryPerBlockOptin
#define GpuFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize

#define GpuErrorT cudaError_t
#define GpuSuccess cudaSuccess
#define GpuGetErrorString cudaGetErrorString
#define GpuGetLastError cudaGetLastError

#elif defined(PSZ_USE_HIP)

// #include <hip/hip_runtime.h>
#include "port/primitives_hip.hh"
#define GpuStreamT hipStream_t
#define GpuStreamCreate hipStreamCreate
#define GpuStreamDestroy hipStreamDestroy

#define GpuDeviceSync hipDeviceSynchronize
#define GpuStreamSync(STREAM) hipStreamSynchronize((hipStream_t)STREAM)

#define GpuMemcpy hipMemcpy
#define GpuMemcpyAsync hipMemcpyAsync
#define GpuMemcpyKind hipMemcpyKind
#define GpuMemcpyH2D hipMemcpyHostToDevice
#define GpuMemcpyH2H hipMemcpyHostToHost
#define GpuMemcpyD2H hipMemcpyDeviceToHost
#define GpuMemcpyD2D hipMemcpyDeviceToDevice

#define GpuMalloc hipMalloc
#define GpuFree hipFree
#define GpuFreeHost hipHostFree

#define GpuGetDevice hipGetDevice
#define GpuSetDevice hipSetDevice
#define GpuDeviceProp hipDeviceProp_t
#define GpuGetDeviceProperties hipGetDeviceProperties
#define GpuDeviceGetAttribute hipDeviceGetAttribute
#define GpuDevicePtr hipDeviceptr_t
#define GpuMemGetAddressRange hipMemGetAddressRange
#define GpuFuncSetAttribute hipFuncSetAttribute
#define GpuFuncAttribute hipFuncAttribute

#define GpuDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define GpuDevAttrMaxSharedMemoryPerBlock hipDeviceAttributeMaxSharedMemoryPerBlock
#define GpuDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeMaxSharedMemoryPerBlock
#define GpuFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

#define GpuErrorT hipError_t
#define GpuSuccess hipSuccess
#define GpuGetErrorString hipGetErrorString
#define GpuGetLastError hipGetLastError

#endif
