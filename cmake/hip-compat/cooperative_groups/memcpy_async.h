// Copyright (c) 2026 Advanced Micro Devices, Inc.
//
// Shim for sources that #include <cooperative_groups/memcpy_async.h>. HIP's
// cooperative-groups support lives in a single header and does not split out a
// memcpy_async sub-header; redirect to it so the include resolves.
#ifndef PSZ_HIP_COMPAT_CG_MEMCPY_ASYNC_H
#define PSZ_HIP_COMPAT_CG_MEMCPY_ASYNC_H
#include <hip/hip_cooperative_groups.h>
#endif
