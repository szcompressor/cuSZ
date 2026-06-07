#ifndef _PORTABLE_MEM_GPU_EVENT_HH
#define _PORTABLE_MEM_GPU_EVENT_HH

#include <cuda_runtime.h>
#include <cupti_activity.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <tuple>

namespace _ptb {

// CUDA event zone: RAII ///////////////////////////////////////////////////////

struct _gpu_event_deleter {
  void operator()(CUevent_st* e) const noexcept
  {
    if (e) cudaEventDestroy(e);
  }
};

using gpu_event = std::unique_ptr<CUevent_st, _gpu_event_deleter>;

inline gpu_event make_gpu_event()
{
  cudaEvent_t           raw = nullptr;
  [[maybe_unused]] auto err = cudaEventCreate(&raw);
  assert(err == cudaSuccess && "cudaEventCreate failed");
  return gpu_event(raw);
}

// RAII cudaEvent runner
struct timer_cuevent {
  float     ms;
  gpu_event e0 = make_gpu_event();
  gpu_event e1 = make_gpu_event();

  void start(cudaStream_t s) { cudaEventRecord(e0.get(), s); }

  double stop_ms(cudaStream_t s)
  {
    cudaEventRecord(e1.get(), s);
    cudaEventSynchronize(e1.get());
    cudaEventElapsedTime(&ms, e0.get(), e1.get());
    return ms;
  }
};

// CUPTI zone //////////////////////////////////////////////////////////////////

// CUPTI kernel timing: sum of on-device kernel durations (vs cudaEvent wall-clock).
// Buffers are app-owned (request=malloc, complete=parse+free).
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

  // register the buffer callbacks once; switches gpu_timer to the CUPTI backend.
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

// user zone //////////////////////////////////////////////////////////////////

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

// related utils ///////////////////////////////////////////////////////////////

template <typename T>
inline std::tuple<size_t, double> bytes_GiBps(size_t len, double ms)  // GiBps
{
  const auto B_to_GiB = 1.0 * 1024 * 1024 * 1024;
  auto       bytes    = len * sizeof(T) * 1.0;
  auto       gibps    = ms > 0 ? bytes / (ms * 1e-3) / B_to_GiB : 0.0;
  return {bytes, gibps};
}

template <typename T>
inline double GiBps(size_t len, double ms)  // GiBps
{
  auto [_, gibps] = bytes_GiBps<T>(len, ms);
  return gibps;
}

}  // namespace _ptb

#endif  // _PORTABLE_MEM_GPU_EVENT_HH
