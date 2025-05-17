#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "detail/compare.hh"

template <typename T>
T* allocate_device_memory(const std::vector<T>& host_data)
{
  T* device_data;
  cudaMalloc(&device_data, host_data.size() * sizeof(T));
  cudaMemcpy(device_data, host_data.data(), host_data.size() * sizeof(T), cudaMemcpyHostToDevice);
  return device_data;
}

template <typename T>
void free_device_memory(T* device_data)
{
  cudaFree(device_data);
}

template <typename T>
void test_find_max_error()
{
  // Generate test data
  const size_t len = 3000;
  std::vector<T> a(len, 1.0f);  // Array of 1.0
  std::vector<T> b(len, 1.0f);  // Array of 1.0
  b[len / 2] = (T)2.0;          // Introduce a difference at one index

  T maxval_cpu;
  size_t maxloc_cpu;
  psz::cppstl::CPU_find_max_error<T>(a.data(), b.data(), len, maxval_cpu, maxloc_cpu);

  T maxval_gpu;
  size_t maxloc_gpu;
  T* d_a = allocate_device_memory(a);
  T* d_b = allocate_device_memory(b);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  psz::module::GPU_find_max_error(d_a, d_b, len, maxval_gpu, maxloc_gpu, stream);

  free_device_memory(d_a);
  free_device_memory(d_b);
  cudaStreamDestroy(stream);

  assert(std::fabs(maxval_cpu - maxval_gpu) < 1e-6);  // ensure the maximum error matches
  assert(maxloc_cpu == maxloc_gpu);                   // ensure the indices match
  assert(len / 2 == maxloc_gpu);                      // ensure the indices match

  std::cout << "test passed: GPU and CPU implementations produce the same results.\n";
  std::cout << "maximum error: " << maxval_cpu << " at index: " << maxloc_cpu << "\n";
}

int main()
{
  try {
    test_find_max_error<float>();
    test_find_max_error<double>();
  }
  catch (const std::exception& e) {
    std::cerr << "test failed: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}