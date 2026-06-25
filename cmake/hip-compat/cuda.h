// Copyright (c) 2026 Advanced Micro Devices, Inc.
//
// Shim placed on the HIP include path so sources that #include <cuda.h> resolve
// against the HIP runtime; the CUDA->HIP translation macros are supplied by the
// force-included psz_hip_compat.h prelude / the cuda_runtime.h shim.
#ifndef PSZ_HIP_COMPAT_CUDA_H
#define PSZ_HIP_COMPAT_CUDA_H
#include <hip/hip_runtime.h>
#endif
