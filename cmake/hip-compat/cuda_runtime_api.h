// Copyright (c) 2026 Advanced Micro Devices, Inc.
//
// Shim so sources that #include <cuda_runtime_api.h> resolve against the HIP
// runtime. The CUDA->HIP translation macros are supplied by the cuda_runtime.h
// shim / the force-included psz_hip_compat.h prelude.
#ifndef PSZ_HIP_COMPAT_CUDA_RUNTIME_API_H
#define PSZ_HIP_COMPAT_CUDA_RUNTIME_API_H
#include <hip/hip_runtime.h>
#endif
