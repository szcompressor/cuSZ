#ifndef _PORTABLE_BACKEND_H
#define _PORTABLE_BACKEND_H

// This file is only for compatibility.
// Compile-time PROPER_RUNTIME and PROPER_EB selection.

#include "c_type.h"

#if defined(PSZ_USE_CUDA)
#define PROPER_RUNTIME _ptb_runtime::CUDA
#define PROPER_EB f8
#elif defined(PSZ_USE_HIP)
#define PROPER_RUNTIME _ptb_runtime::HIP
#define PROPER_EB f8
#elif defined(PSZ_USE_1API)
#define PROPER_RUNTIME _ptb_runtime::ONEAPI
#define PROPER_EB f4
#endif

#endif /* _PORTABLE_BACKEND_H */
