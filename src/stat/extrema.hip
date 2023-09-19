#include <hip/hip_runtime.h>
#include "cusz/type.h"
#include "stat/compare.cu_hip.hh"
#include "utils/err.hh"
#include "port.hh"
// definitions
#include "detail/extrema.cu_hip.inl"

template void psz::cu_hip::extrema<f4>(f4* d_ptr, szt len, f4 res[4]);
template void psz::cu_hip::extrema<f8>(f8* d_ptr, szt len, f8 res[4]);