// Templated CPU/GPU sanity for psz::cuda::GPU_identical (small len, fixed perturb position).
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "compare.hh"
#include "mem/cxx_backends.h"
#include "mem/cxx_smart_ptr.h"

template <typename T>
bool check(cudaStream_t stream, T test_val, T test_delta)
{
  constexpr size_t len      = 1027;
  constexpr size_t sizeof_T = sizeof(T);
  constexpr size_t bytes    = len * sizeof_T;

  std::vector<T> h1(len, test_val), h2(len, test_val);
  h2[1025] = test_val + test_delta;

  auto d1 = MAKE_UNIQUE_DEVICE(uint8_t, bytes);
  auto d2 = MAKE_UNIQUE_DEVICE(uint8_t, bytes);
  memcpy_allkinds<H2D>(d1.get(), (uint8_t*)h1.data(), bytes);
  memcpy_allkinds<H2D>(d2.get(), (uint8_t*)h2.data(), bytes);

  bool cpu_ok = psz::cppstl::CPU_identical(h1.data(), h2.data(), sizeof_T, len);
  bool gpu_ok = psz::cuda::GPU_identical(d1.get(), d2.get(), sizeof_T, len, stream);
  return cpu_ok == gpu_ok;
}

int main()
{
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  bool ok = true;
  ok &= check<float>(stream, 1.0f, 1.0f);
  ok &= check<float>(stream, 1.1f, 1.0f);
  ok &= check<float>(stream, 1.0f, 0.0f);
  ok &= check<float>(stream, 1.1f, 0.0f);
  ok &= check<uint8_t>(stream, 1, 1);
  ok &= check<uint8_t>(stream, 1, 0);

  cudaStreamDestroy(stream);
  return ok ? 0 : 1;
}
