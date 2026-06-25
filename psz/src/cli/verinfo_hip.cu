// Copyright (c) 2026 Advanced Micro Devices, Inc.
//
// \file verinfo_hip.cu
// \author Jeff Daily <jeff.daily@amd.com>
// \brief HIP/ROCm implementation of the CLI version-info queries.
//
// On the HIP backend this replaces verinfo.cu (NVML driver query) and
// verinfo_nv.cu (CUDA driver-API deviceQuery). The function names mirror the
// CUDA backend (print_NVCC_ver, print_CUDA_driver, print_NVIDIA_driver,
// CUDA_devices) so the shared context.cc version banner is unchanged; here they
// report the HIP/ROCm toolchain, driver, and device properties.

#include <hip/hip_runtime.h>

#include <cstdio>

#include "cli/verinfo.h"

void print_NVCC_ver()
{
  printf(
      "- HIP: %s.%s.%s\n",  //
      STRINGIZE_VALUE_OF(HIP_VERSION_MAJOR), STRINGIZE_VALUE_OF(HIP_VERSION_MINOR),
      STRINGIZE_VALUE_OF(HIP_VERSION_PATCH));
}

int print_CUDA_driver()
{
  int driver_version = 0;
  std::printf("- HIP driver: ");
  if (hipDriverGetVersion(&driver_version) != hipSuccess) {
    std::printf("(failed to get driver version)\n");
    return 1;
  }
  int major_version = driver_version / 10000000;
  int minor_version = (driver_version % 10000000) / 100000;
  std::printf("%d.%d\n", major_version, minor_version);
  return 0;
}

int print_NVIDIA_driver()
{
  int runtime_version = 0;
  if (hipRuntimeGetVersion(&runtime_version) != hipSuccess) {
    std::printf("- ROCm runtime: (failed to query)\n");
    return 1;
  }
  int major_version = runtime_version / 10000000;
  int minor_version = (runtime_version % 10000000) / 100000;
  std::printf("- ROCm runtime: %d.%d\n", major_version, minor_version);
  return 0;
}

void CUDA_devices()
{
  int device_count = 0;
  hipGetDeviceCount(&device_count);

  if (device_count == 0) { printf("0 devices detected\n"); }
  else {
    printf("%d HIP device(s):\n", device_count);
  }

  for (auto dev = 0; dev < device_count; ++dev) {
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, dev);

    int  memClockKHz = deviceProp.memoryClockRate;
    auto membw_GiBps = membw_base1024(deviceProp.memoryBusWidth, memClockKHz * 1e3);
    auto membw_GBps  = membw_base1000(deviceProp.memoryBusWidth, memClockKHz * 1e3);

    printf("- %s (%s)\n", deviceProp.name, deviceProp.gcnArchName);
    printf(
        "  - %d compute units; warp size: %d\n", deviceProp.multiProcessorCount,
        deviceProp.warpSize);
    printf(
        "  - global VRAM: %.0f MB (theoretically) at \n",
        (float)deviceProp.totalGlobalMem / 1048576.0f);
    printf("    %.1f GiB/s (base-1024) or %.1f GB/s (base-1000)\n", membw_GiBps, membw_GBps);
    printf("  - L2 cache: %d bytes\n", deviceProp.l2CacheSize);
    printf(
        "  - per-block/CU total shared memory: %zu/%zu bytes\n", deviceProp.sharedMemPerBlock,
        deviceProp.sharedMemPerMultiprocessor);
    printf(
        "  - per-block/CU max thread count: %d/%d\n", deviceProp.maxThreadsPerBlock,
        deviceProp.maxThreadsPerMultiProcessor);
    printf(
        "  - max thread-block dim (x,y,z): (%d, %d, %d)\n", deviceProp.maxThreadsDim[0],
        deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("  - per-block total registers count: %d\n", deviceProp.regsPerBlock);
  }
}
