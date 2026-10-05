#ifndef _PORTABLE_MEM_GPU_TIMER_HH
#define _PORTABLE_MEM_GPU_TIMER_HH

#include <cuda_runtime.h>
#include <cupti_activity.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>

#include "mem/gpu_event.hh"

namespace _ptb {

struct timer_cupti {
  // CUpti_ActivityKernel* v5 onward shares the start/end layout.
#if CUPTI_API_VERSION >= 18  // CUDA 12.x
  using AK = CUpti_ActivityKernel11;
#elif CUPTI_API_VERSION >= 17  // CUDA 11.6–11.8
  using AK = CUpti_ActivityKernel9;
#elif CUPTI_API_VERSION >= 15  // CUDA 11.0–11.5
  using AK = CUpti_ActivityKernel8;
#elif CUPTI_API_VERSION >= 13  // CUDA 10.x
  using AK = CUpti_ActivityKernel6;
#else
  using AK = CUpti_ActivityKernel5;
#endif

  static inline bool                  active = false;
  static inline std::atomic<uint64_t> kernel_ns{0};

  static void CUPTIAPI buf_requested(uint8_t** buf, size_t* sz, size_t* max_rec)
  {
    *sz      = 1u << 20;  // 1 MiB
    *buf     = (uint8_t*)malloc(*sz);
    *max_rec = 0;
  }

  static void CUPTIAPI buf_completed(CUcontext, uint32_t, uint8_t* buf, size_t, size_t valid)
  {
    CUpti_Activity* rec = nullptr;
    while (cuptiActivityGetNextRecord(buf, valid, &rec) == CUPTI_SUCCESS) {
      if (rec->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL or
          rec->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
        auto* k = (AK*)rec;
        if (k->start != 0 and k->end >= k->start) kernel_ns += k->end - k->start;
      }
    }
    free(buf);
  }

  static void enable()
  {
    cuptiActivityRegisterCallbacks(buf_requested, buf_completed);
    active = true;
  }

  void start(cudaStream_t)
  {
    kernel_ns = 0;
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  }

  double stop_ms(cudaStream_t s)
  {
    cudaStreamSynchronize(s);
    cuptiActivityFlushAll(0);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    return kernel_ns.load() * 1e-6;
  }
};

struct gpu_timer {
  timer_cuevent ev_timer;
  timer_cupti   cupti_timer;

  void start(cudaStream_t s)
  {
    if (timer_cupti::active)
      cupti_timer.start(s);
    else
      ev_timer.start(s);
  }

  double stop_ms(cudaStream_t s)
  {
    if (timer_cupti::active)
      return cupti_timer.stop_ms(s);
    else
      return ev_timer.stop_ms(s);
  }
};

}  // namespace _ptb

#endif  // _PORTABLE_MEM_GPU_TIMER_HH
