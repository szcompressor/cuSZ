#include <cuda_runtime.h>

#include "cusz/type.h"
#include "stat/compare.hh"

namespace psz {

__global__ void GPU_CUHIP_identical(
    void* d1, void* d2, size_t sizeof_T, size_t const len, uint32_t* result)
{
  uint8_t* data1 = (uint8_t*)d1;
  uint8_t* data2 = (uint8_t*)d2;

  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  // loop over uints of uint32_t
  for (size_t i = idx * sizeof(uint32_t); i < len * sizeof_T;
       i += sizeof(uint32_t) * blockDim.x * gridDim.x) {
    // periodically check the global result
    if (not(*result)) return;

    // check if this is the last, partial chunk
    if (i + sizeof(uint32_t) <= len * sizeof_T) {
      // process full uint32_t uints
      uint32_t* packed_data1 = (uint32_t*)(data1 + i);
      uint32_t* packed_data2 = (uint32_t*)(data2 + i);
      if (*packed_data1 != *packed_data2) {
        atomicExch(result, (uint32_t)false);
        return;
      }
    }
    else {
      // handle remaining bytes as individual uint8_t comparisons
      for (size_t j = i; j < len * sizeof_T; ++j) {
        uint8_t* byte_data1 = (uint8_t*)data1;
        uint8_t* byte_data2 = (uint8_t*)data2;
        if (byte_data1[j] != byte_data2[j]) {
          atomicExch(result, (uint32_t)false);
          return;
        }
      }
    }
  }
}

}  // namespace psz

namespace psz::module {

bool GPU_identical(void* d1, void* d2, size_t sizeof_T, size_t const len, void* stream)
{
  uint32_t* result;
  cudaMallocManaged(&result, sizeof(uint32_t));
  *result = true;

  auto block_size = 256;
  auto grid_size = (len * sizeof_T + block_size - 1) / block_size;

  GPU_CUHIP_identical<<<block_size, grid_size, 0, (cudaStream_t)stream>>>(
      d1, d2, sizeof_T, len, result);
  cudaStreamSynchronize((cudaStream_t)stream);

  bool host_result = (bool)*result;
  cudaFree(result);
  return host_result;
}

}  // namespace psz::module