#if defined(PSZ_USE_CUDA)

#include <cuda_runtime.h>
#include "port/cuda_indication.hh"

#elif defined(PSZ_USE_HIP)

#include <hip/hip_runtime.h>
#include "port/hip_indication.hh"
#include "port/hip_primitives.hh"
#include "port/hip_suppress_warning.hh"

#elif defined(PSZ_USE_1API)

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "port/dpcpp_indication.hh"

#endif

#include "port/proper_backend.hh"