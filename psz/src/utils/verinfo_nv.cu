/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* This sample queries the properties of the CUDA devices present in the system
 * via CUDA Runtime API. */

// modified from deviceQuery.cpp of CUDA Samples

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

#include "utils/verinfo.h"

void print_NVCC_ver()
{
  printf(
      "- nvcc: %s.%s.%s\n",  //
      STRINGIZE_VALUE_OF(__CUDACC_VER_MAJOR__),
      STRINGIZE_VALUE_OF(__CUDACC_VER_MINOR__),
      STRINGIZE_VALUE_OF(__CUDACC_VER_BUILD__));
}

int print_CUDA_driver()
{
  CUresult result;
  int driver_version = 0;

  std::cout << "- CUDA driver: ";

  result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    const char* error_str;
    cuGetErrorString(result, &error_str);
    std::cerr << "(failed to initialize: " << error_str << ")" << std::endl;
    return 1;
  }

  // Get the CUDA driver version
  result = cuDriverGetVersion(&driver_version);
  if (result != CUDA_SUCCESS) {
    const char* error_str;
    cuGetErrorString(result, &error_str);
    std::cerr << "(failed to get driver version: " << error_str << ")"
              << std::endl;
    return 1;
  }

  int major_version = driver_version / 1000;
  int minor_version = (driver_version % 1000) / 10;
  std::cout << major_version << "." << minor_version << std::endl;

  return 0;
}

template <class T>
inline void getCudaAttribute(
    T* attribute, CUdevice_attribute device_attribute, int device)
{
  cuDeviceGetAttribute(attribute, device_attribute, device);
}

void CUDA_devices()
{
  int device_count = 0;
  cuDeviceGetCount(&device_count);

  if (device_count == 0) { printf("0 devices detected\n"); }
  else {
    printf("%d CUDA device(s):\n", device_count);
  }

  for (auto dev = 0; dev < device_count; ++dev) {
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    auto membw_GiBps = membw_base1024(
        deviceProp.memoryBusWidth, deviceProp.memoryClockRate * 1e3);
    auto membw_GBps = membw_base1000(
        deviceProp.memoryBusWidth, deviceProp.memoryClockRate * 1e3);

    printf("- %s\n", deviceProp.name);
    printf(
        "  - %d multiprocessors; SM capability: %d.%d; warp size: %d\n",
        deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor,
        deviceProp.warpSize);
    printf(
        "  - global VRAM: %.0f MB (theoretically) at \n",
        (float)deviceProp.totalGlobalMem / 1048576.0f);
    printf(
        "    %.1f GiB/s (base-1024) or %.1f GB/s (base-1000)\n", membw_GiBps,
        membw_GBps);
    printf("  - L2 cache: %d bytes\n", deviceProp.l2CacheSize);
    printf(
        "  - per-block/SM total shared memory: %zu/%zu bytes\n",
        deviceProp.sharedMemPerBlock, deviceProp.sharedMemPerMultiprocessor);
    // printf("  - total constant memory: %zu bytes\n",
    // deviceProp.totalConstMem);
    printf(
        "  - per-block/SM max thread count: %d/%d\n",
        deviceProp.maxThreadsPerBlock, deviceProp.maxThreadsPerMultiProcessor);
    printf(
        "  - max thread-block dim (x,y,z): (%d, %d, %d)\n",
        deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
        deviceProp.maxThreadsDim[2]);
    printf(
        "  - per-block total registers count: %d\n", deviceProp.regsPerBlock);
  }
}