// Copyright (c) 2026 Advanced Micro Devices, Inc.
//
// HIP/ROCm backend compatibility prelude for cuSZ.
//
// This header is force-included (compiler -include) into every translation
// unit compiled for the HIP backend. The kernel and host sources are written
// in CUDA spelling and reused as-is for HIP (single-source, no .hip mirrors);
// this prelude pulls in the HIP runtime and the CUDA->HIP translation macros so
// those sources compile unchanged against ROCm. It is never on the CUDA or SYCL
// include path, so those backends are unaffected.

#ifndef PSZ_HIP_COMPAT_H
#define PSZ_HIP_COMPAT_H

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include "macro/c_cu2hip_0_translation.h"
#include "macro/c_cu2hip_1_fix_primitives.h"
#include "macro/c_cu2hip_2_suppress_warning.h"

#endif /* PSZ_HIP_COMPAT_H */
