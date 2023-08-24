#if defined(PSZ_USE_CUDA)

// #include <cuda_runtime.h>

#elif defined(PSZ_USE_HIP)

// #include <hip/hip_runtime.h>
#include "port/primitives_hip.hh"

#endif
