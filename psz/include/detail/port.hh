#if defined(PSZ_USE_CUDA)

#include <cuda_runtime.h>

#define cudaMallocShared(...) cudaMallocManaged(__VA_ARGS__)

#define PROPER_RUNTIME psz_runtime::CUDA
#define PROPER_EB f8

#elif defined(PSZ_USE_HIP)

#include <hip/hip_runtime.h>

#include "macro/c_cu2hip_0_translation.h"
#include "macro/c_cu2hip_1_fix_primitives.h"
#include "macro/c_cu2hip_2_suppress_warning.h"

#define PROPER_RUNTIME psz_runtime::HIP
#define PROPER_EB f8

#elif defined(PSZ_USE_1API)

#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#define PROPER_RUNTIME psz_runtime::ONEAPI
#define PROPER_EB f4

#endif
