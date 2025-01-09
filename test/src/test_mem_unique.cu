#include <cuda_runtime.h>

#include <cassert>
#include <iostream>

#include "mem/cxx_backends.h"  // Assuming this is your provided file

constexpr size_t len = 1024;
constexpr float scalar = 5.0f;
constexpr size_t bytes = len * sizeof(float);

constexpr dim3 block(256);
constexpr dim3 grid((len - 1) / block.x + 1);

__global__ void add_scalar(float* arr, size_t len, float scalar)
{
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    arr[idx] += scalar;
    // printf("arr[%d] = %f\n", idx, arr[idx]);
  }
}

int test_cuda_unique_ptr()
{
  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  float* d_array_native = nullptr;
  float* h_array_native = new float[len];

  cudaMalloc(&d_array_native, bytes);
  cudaMemset(d_array_native, 0, bytes);

  add_scalar<<<grid, block>>>(d_array_native, len, scalar);
  cudaDeviceSynchronize();

  cudaMemcpy(h_array_native, d_array_native, bytes, cudaMemcpyDeviceToHost);

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  auto d_array_smart = GPU_make_unique(malloc_d<float>(len), GPU_deleter_d());
  auto h_array_smart = GPU_make_unique(malloc_h<float>(len), GPU_deleter_h());

  add_scalar<<<grid, block>>>(d_array_smart.get(), len, scalar);
  cudaDeviceSynchronize();

  memcpy_allkinds<D2H>(h_array_smart.get(), d_array_smart.get(), len);

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  bool success = true;
  for (size_t i = 0; i < len; i++) {
    if (h_array_native[i] != h_array_smart[i]) {
      success = false;
      printf("(native[%d] = %f) != (smart[%d] = %f)\n", i, h_array_native[i], i, h_array_smart[i]);
      break;
    }
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  cudaFree(d_array_native);
  delete[] h_array_native;

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  if (success) {
    std::cout << "passed: GPU_unique_ptr matches native impl." << std::endl;
    return 0;
  }
  else {
    std::cerr << "FAILED: results mismatch." << std::endl;
    return 1;
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
}

int main() { return test_cuda_unique_ptr(); }