#ifndef _PORTABLE_MEM_GPU_STREAM_HH
#define _PORTABLE_MEM_GPU_STREAM_HH

// RAII wrapper around cudaStream_t.
//
// Usage:
//   auto stream = _ptb::make_gpu_stream();
//   kernel<<<..., stream.get()>>>(...);
//   // stream is destroyed automatically at scope exit
//
// Design notes:
//   - cudaStream_t is `struct CUstream_st*`, so it fits std::unique_ptr.
//   - The cudaStreamCreate return value is checked at construction (assert).
//     Stream creation can fail under driver/OOM conditions and we want to
//     fail fast rather than silently produce a null stream.
//   - cudaStreamDestroy is fire-and-forget in the deleter: the call is
//     asynchronous and reporting destruction errors is rarely actionable.
//   - assert() compiles out under NDEBUG; a Release build trusts the driver.

#include <cuda_runtime.h>

#include <cassert>
#include <memory>
#include <type_traits>

namespace _ptb {

using _gpu_stream_elem = std::remove_pointer<cudaStream_t>::type;

struct gpu_stream_deleter {
  void operator()(_gpu_stream_elem* s) const noexcept
  {
    if (s) cudaStreamDestroy(s);
  }
};

using gpu_stream = std::unique_ptr<_gpu_stream_elem, gpu_stream_deleter>;

inline gpu_stream make_gpu_stream()
{
  cudaStream_t          raw = nullptr;
  [[maybe_unused]] auto err = cudaStreamCreate(&raw);
  assert(err == cudaSuccess && "cudaStreamCreate failed");
  return gpu_stream(raw);
}

}  // namespace _ptb

#endif  // _PORTABLE_MEM_GPU_STREAM_HH
