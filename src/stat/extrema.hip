#include <hip/hip_runtime.h>
#include "cusz/type.h"
#include "stat/compare_gpu.hh"
#include "utils/err.hh"
#include "port.hh"
// definitions
#include "detail/extrema_g.inl"

template void psz::cuda_hip_compat::extrema<f4>(f4* d_ptr, szt len, f4 res[4]);
template void psz::cuda_hip_compat::extrema<f8>(f8* d_ptr, szt len, f8 res[4]);