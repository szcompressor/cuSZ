#include <cuda_runtime.h>

#include <numeric>
#include <type_traits>

namespace psz {

template <typename T, size_t BlockSize>
__global__ void KERNEL_CUHIP_find_max_errors(
    T* a, T* b, size_t len, T* block_max_errors, size_t* block_max_indices)
{
  __shared__ T s_errors[BlockSize];
  __shared__ size_t s_indices[BlockSize];
  auto tid = threadIdx.x;
  auto gD_id = blockIdx.x * blockDim.x + tid;

  // initialize shared memory
  s_errors[tid] = std::numeric_limits<T>::lowest();
  s_indices[tid] = 0;
  __syncthreads();

  // calculate error for this thread
  if (gD_id < len) {
    if constexpr (std::is_same_v<T, float>) {
      T error = fabsf(a[gD_id] - b[gD_id]);
      s_errors[tid] = error;
      s_indices[tid] = gD_id;
    }
    else {
      T error = fabs(a[gD_id] - b[gD_id]);
      s_errors[tid] = error;
      s_indices[tid] = gD_id;
    }
  }
  else {
    s_errors[tid] = std::numeric_limits<T>::lowest();
  }
  __syncthreads();

  // reduce within the block
  for (auto stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      if (s_errors[tid] < s_errors[tid + stride]) {
        s_errors[tid] = s_errors[tid + stride];
        s_indices[tid] = s_indices[tid + stride];
      }
    }
    __syncthreads();
  }

  // write back
  if (tid == 0) {
    block_max_errors[blockIdx.x] = s_errors[0];
    block_max_indices[blockIdx.x] = s_indices[0];
  }
}

}  // namespace psz

namespace psz::module {

template <typename T>
void GPU_find_max_error(T* a, T* b, size_t const len, T& maxval, size_t& maxloc, void* stream)
{
  constexpr size_t threads_per_block = 256;
  const size_t blocks = (len + threads_per_block - 1) / threads_per_block;

  T* d_max_errors;
  size_t* d_max_indices;
  cudaMallocManaged(&d_max_errors, blocks * sizeof(T));
  cudaMallocManaged(&d_max_indices, blocks * sizeof(size_t));

  psz::KERNEL_CUHIP_find_max_errors<T, threads_per_block>       //
      <<<blocks, threads_per_block, 0, (cudaStream_t)stream>>>  //
      (a, b, len, d_max_errors, d_max_indices);
  cudaStreamSynchronize((cudaStream_t)stream);

  maxval = std::numeric_limits<T>::lowest();
  maxloc = 0;
  for (size_t i = 0; i < blocks; ++i) {
    if (d_max_errors[i] > maxval) {
      maxval = d_max_errors[i];
      maxloc = d_max_indices[i];
    }
  }

  cudaFree(d_max_errors);
  cudaFree(d_max_indices);
}

}  // namespace psz::module