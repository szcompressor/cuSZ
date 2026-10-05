#ifndef _PORTABLE_MEM_GPU_EVENT_HH
#define _PORTABLE_MEM_GPU_EVENT_HH

#include <cuda_runtime.h>

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
