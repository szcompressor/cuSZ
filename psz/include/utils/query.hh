#ifndef C5CF22D0_3237_4B41_9907_531D96EAA7F0
#define C5CF22D0_3237_4B41_9907_531D96EAA7F0

#include "busyheader.hh"
#include "cusz/type.h"
#include "port.hh"
//
#include "query/query_cpu.hh"
#define CPU_QUERY cpu_diagnostics::get_cpu_properties();
//
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
#include "query/query_dev.cu_hip.hh"

#define GPU_QUERY cu_hip_diagnostics::get_device_property();

#elif defined(PSZ_USE_1API)
#include "query/query_dev.l0.hh"

// #define GPU_QUERY \
//   l0_diagnostics::show_device(dpct::get_current_device().default_queue());
#define GPU_QUERY \
  l0_diagnostics::show_device();

#endif

#endif /* C5CF22D0_3237_4B41_9907_531D96EAA7F0 */
