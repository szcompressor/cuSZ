// author: Boyuan Zhang
// refactor: Jiannan Tian

#include <cstddef>
#include <cstdint>

namespace fzgpu {

__global__ void KERNEL_CUHIP_fz_fused_decode(
    uint32_t* in_archive, uint32_t* in_bitflag_array, uint32_t* in_start_pos,
    uint32_t* out_decoded, size_t const decoded_len)
{
  // allocate shared byte flag array
  __shared__ uint32_t s_data_chunk[32][33];
  __shared__ uint16_t s_byteflag_array[257];
  __shared__ uint32_t s_start_pos;

  /* start of resettig shared memory */
  s_data_chunk[threadIdx.y][threadIdx.x] = 0;
  auto tid = threadIdx.y * blockDim.x + threadIdx.x;
  if (tid < 257) s_byteflag_array[tid] = 0;
  __syncthreads();
  /* end of resettig shared memory */

  // there are 32 x 32 uint32_t in_data this data chunk
  auto bid = blockIdx.x;

  // transfer bit flag array to byte flag array
  auto bigflag = 0u;
  if (threadIdx.x < 8 and threadIdx.y == 0) {
    bigflag = in_bitflag_array[bid * 8 + threadIdx.x];
#pragma unroll 32
    for (int tmpInd = 0; tmpInd < 32; tmpInd++) {
      s_byteflag_array[threadIdx.x * 32 + tmpInd] =
          (bigflag & (1U << tmpInd)) > 0;
    }
  }
  __syncthreads();

  auto prefix_sum_offset = 1u;
  constexpr auto block_size = 256u;

  // prefix summation, up-sweep
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

  // initialize the shared memory to all 0
  s_data_chunk[threadIdx.y][threadIdx.x] = 0;
  __syncthreads();

  // get the start position
  if (threadIdx.x == 0 and threadIdx.y == 0) {
    s_start_pos = in_start_pos[bid];
  }
  __syncthreads();

  // write back shuffled data to shared mem
  auto byteflag_ind = tid / 4;
  if (s_byteflag_array[byteflag_ind + 1] != s_byteflag_array[byteflag_ind]) {
    s_data_chunk[threadIdx.x][threadIdx.y] =
        in_archive[s_start_pos + s_byteflag_array[byteflag_ind] * 4 + tid % 4];
  }
  __syncthreads();

  // store the corresponding uint32 to the register buffer
  auto buffer = s_data_chunk[threadIdx.y][threadIdx.x];
  __syncthreads();

  // bitshuffle (reverse)
#pragma unroll 32
  for (auto i = 0; i < 32; i++) {
    s_data_chunk[threadIdx.y][i] =
        __ballot_sync(0xFFFFFFFFU, buffer & (1U << i));
  }
  __syncthreads();

  // write back to global memory
  auto gD_id = tid + bid * (blockDim.x * blockDim.y);
  if (gD_id < decoded_len)
    out_decoded[gD_id] = s_data_chunk[threadIdx.y][threadIdx.x];
}

}  // namespace fzgpu