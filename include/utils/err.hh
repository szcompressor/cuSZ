#ifndef AE6DCA2E_F19B_41DB_80CB_11230E548F92
#define AE6DCA2E_F19B_41DB_80CB_11230E548F92

#include "busyheader.hh"
#include "port.hh"

#if defined(PSZ_USE_CUDA)

#include <cuda_runtime.h>
#include "err/err.cu_hip.hh"

#elif defined(PSZ_USE_HIP)

#include <hip/hip_runtime.h>
#include "err/err.cu_hip.hh"

#elif defined(PSZ_USE_1API)

#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#include "err/err.dp.hh"

#endif

#endif /* AE6DCA2E_F19B_41DB_80CB_11230E548F92 */
