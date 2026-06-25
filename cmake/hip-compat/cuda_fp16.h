// Copyright (c) 2026 Advanced Micro Devices, Inc.
//
// Shim so sources that #include <cuda_fp16.h> resolve against HIP's half type.
#ifndef PSZ_HIP_COMPAT_CUDA_FP16_H
#define PSZ_HIP_COMPAT_CUDA_FP16_H
#include <hip/hip_fp16.h>
#endif
