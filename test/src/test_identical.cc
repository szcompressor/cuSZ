#include <cuda_runtime.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

#include "detail/compare.hh"

void* allocate_device_memory(const void* host_data, size_t size)
{
  void* device_data;
  cudaMalloc(&device_data, size);
  cudaMemcpy(device_data, host_data, size, cudaMemcpyHostToDevice);
  return device_data;
}

void deallocate_device_memory(void* device_data) { cudaFree(device_data); }

template <typename T>
void test_identical_functions(float test_val, float test_delta)
{
  size_t len = 1027;
  size_t sizeof_T = sizeof(T);
  std::vector<T> data1(len, test_val);
  std::vector<T> data2(len, test_val);
  data2[1025] = test_val + test_delta;

  bool cpu_result_identical =
      psz::cppstl::CPU_identical(data1.data(), data2.data(), sizeof_T, len);

  void* d1 = allocate_device_memory(data1.data(), len * sizeof_T);
  void* d2 = allocate_device_memory(data2.data(), len * sizeof_T);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  bool gpu_result_identical = psz::module::GPU_identical(d1, d2, sizeof_T, len, stream);

  deallocate_device_memory(d1);
  deallocate_device_memory(d2);
  cudaStreamDestroy(stream);

  // Compare results
  assert(cpu_result_identical == gpu_result_identical);

  // Print the result of the test
  if (cpu_result_identical == gpu_result_identical) {
    std::cout << "Test passed: CPU and GPU results match.\n";
  }
  else {
    std::cerr << "Test failed: CPU and GPU results do not match.\n";
    std::exit(EXIT_FAILURE);
  }
}

int main()
{
  std::cout << "testing <float>(1.0f, delta=1.0f)\n";
  test_identical_functions<float>(1.0f, 1.0f);
  std::cout << "testing <float>(1.1f, delta=1.0f)\n";
  test_identical_functions<float>(1.1, 1.0);
  std::cout << "testing <float>(1.1f, delta=0.0)\n";
  test_identical_functions<float>(1.0, 0.0);
  std::cout << "testing <float>(1.1f, delta=0.0)\n";
  test_identical_functions<float>(1.1, 0.0);
  std::cout << "testing <uint8_t>(1, delta=1)\n";
  test_identical_functions<uint8_t>(1, 1);
  std::cout << "testing <uint8_t>(1, delta=0)\n";
  test_identical_functions<uint8_t>(1, 0);

  return 0;
}