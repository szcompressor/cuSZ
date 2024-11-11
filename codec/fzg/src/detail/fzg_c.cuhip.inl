// author: Boyuan Zhang
// refactor: Jiannan Tian

#include <cstddef>
#include <cstdint>

namespace fzgpu {

__global__ void KERNEL_CUHIP_fz_fused_encode(
    uint32_t const* __restrict__ in_data, size_t data_len,
    uint32_t* space_offset_counter, uint32_t* out_bitflag_array,
    uint32_t* out_start_position, uint32_t* __restrict__ out_comp,
    uint32_t* comp_len)
{
  // 32 x 32 data chunk size with one padding for each row, overall 4096 bytes
  // per chunk
  __shared__ uint32_t s_data_chunk[32][33];
  __shared__ uint16_t s_byteflag_array[257];
  __shared__ uint32_t s_bitflag_array[8];
  __shared__ uint32_t s_start_position;

  /* start of resettig shared memory */
  s_data_chunk[threadIdx.y][threadIdx.x] = 0;
  auto tid = threadIdx.y * blockDim.x + threadIdx.x;
  if (tid < 257) s_byteflag_array[tid] = 0;
  if (threadIdx.y == 0 and threadIdx.x < 8) s_bitflag_array[threadIdx.x] = 0;
  __syncthreads();
  /* end of resettig shared memory */

  auto byteflag = 0u;
  auto v = 0u;
  auto gD_id = tid + blockIdx.x * (blockDim.x * blockDim.y);
  if (gD_id < data_len) v = in_data[gD_id];
  __syncthreads();

#pragma unroll 32
  for (auto i = 0; i < 32; i++) {
    s_data_chunk[threadIdx.y][i] = __ballot_sync(0xFFFFFFFFU, v & (1U << i));
  }
  __syncthreads();

  // generate s_byteflag_array
  if (threadIdx.x < 8) {
#pragma unroll 4
    for (auto i = 0; i < 4; i++) {
      byteflag |= s_data_chunk[threadIdx.x * 4 + i][threadIdx.y];
    }
    s_byteflag_array[threadIdx.y * 8 + threadIdx.x] = byteflag > 0;
  }
  __syncthreads();

  // generate bigflag_array
  uint32_t buffer;
  if (threadIdx.y < 8) {
    buffer = s_byteflag_array[threadIdx.y * 32 + threadIdx.x];
    s_bitflag_array[threadIdx.y] = __ballot_sync(0xFFFFFFFFU, buffer);
  }
  __syncthreads();

  // write back bigflag_array to global memory
  if (threadIdx.x < 8 and threadIdx.y == 0) {
    out_bitflag_array[blockIdx.x * 8 + threadIdx.x] =
        s_bitflag_array[threadIdx.x];
  }

  constexpr auto block_size = 256u;

  // prefix summation, up-sweep
  auto prefix_sum_offset = 1u;
#pragma unroll 8
  for (auto d = 256 >> 1; d > 0; d = d >> 1) {
    if (tid < d) {
      auto ai = prefix_sum_offset * (2 * tid + 1) - 1;
      auto bi = prefix_sum_offset * (2 * tid + 2) - 1;
      s_byteflag_array[bi] += s_byteflag_array[ai];
    }
    __syncthreads();
    prefix_sum_offset *= 2;
  }

  // clear the last element
  if (threadIdx.x == 0 and threadIdx.y == 0) {
    s_byteflag_array[block_size] = s_byteflag_array[block_size - 1];
    s_byteflag_array[block_size - 1] = 0;
  }
  __syncthreads();

  // prefix summation, down-sweep
#pragma unroll 8
  for (auto d = 1; d < 256; d *= 2) {
    prefix_sum_offset >>= 1;
    if (tid < d) {
      auto ai = prefix_sum_offset * (2 * tid + 1) - 1;
      auto bi = prefix_sum_offset * (2 * tid + 2) - 1;

      auto t = s_byteflag_array[ai];
      s_byteflag_array[ai] = s_byteflag_array[bi];
      s_byteflag_array[bi] += t;
    }
    __syncthreads();
  }

  // use atomicAdd to reserve a space for compressed data chunk
  if (threadIdx.x == 0 and threadIdx.y == 0) {
    s_start_position =
        atomicAdd(space_offset_counter, s_byteflag_array[block_size] * 4);
    out_start_position[blockIdx.x] = s_start_position;
    comp_len[blockIdx.x] = s_byteflag_array[block_size];
  }
  __syncthreads();

  // write back the compressed data based on the s_start_position
  auto flag_index = (int)floorf(tid / 4);
  if (s_byteflag_array[flag_index + 1] != s_byteflag_array[flag_index]) {
    out_comp[s_start_position + s_byteflag_array[flag_index] * 4 + tid % 4] =
        s_data_chunk[threadIdx.x][threadIdx.y];
  }
}

}  // namespace fzgpu