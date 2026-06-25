// Copyright (c) 2026 Advanced Micro Devices, Inc.
//
// Shim placed on the HIP include path so sources that #include <cuda_runtime.h>
// (kernel .cu TUs and host .cc TUs alike) resolve against the HIP runtime and
// pick up the CUDA->HIP translation macros. Device TUs additionally get the
// force-included psz_hip_compat.h prelude, but a plain host .cc that only
// includes <cuda_runtime.h> is fully covered by this shim alone.
#ifndef PSZ_HIP_COMPAT_CUDA_RUNTIME_H
#define PSZ_HIP_COMPAT_CUDA_RUNTIME_H

#include <hip/hip_runtime.h>

#include "macro/c_cu2hip_0_translation.h"
#include "macro/c_cu2hip_1_fix_primitives.h"
#include "macro/c_cu2hip_2_suppress_warning.h"

#endif
