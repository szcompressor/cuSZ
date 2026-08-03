// Author: Cody Rivera (cjrivera1@crimson.ua.edu), Megan Hickman Fulp (mlhickm@g.clemson.edu)
// Fast histogramming from [Gómez-Luna et al. 2013]

#include <cstdio>
#include <limits>

#include "kernel.hh"
#include "utils/timer.hh"

#define MIN(a, b) ((a) < (b)) ? (a) : (b)
const static unsigned int WARP_SIZE = 32;

#define tix threadIdx.x
#define tiy threadIdx.y
#define tiz threadIdx.z
#define bix blockIdx.x
#define biy blockIdx.y
#define biz blockIdx.z
#define bdx blockDim.x
#define bdy blockDim.y
#define bdz blockDim.z

namespace psz {

/* Copied from J. Gomez-Luna et al. */
template <typename T, typename FREQ>
__global__ void KCU_p2013Histogram(
    T* in_data, size_t const data_len, FREQ* out_bins, uint16_t const bins_len,
    uint16_t const repeat)
{
  extern __shared__ int Hs[/*(bins_len + 1) * repeat*/];

  const unsigned int warp_id = (int)(tix / WARP_SIZE);
  const unsigned int lane = tix % WARP_SIZE;
  const unsigned int warps_block = bdx / WARP_SIZE;
  const unsigned int off_rep = (bins_len + 1) * (tix % repeat);
  const unsigned int begin = (data_len / warps_block) * warp_id + WARP_SIZE * blockIdx.x + lane;
  unsigned int end = (data_len / warps_block) * (warp_id + 1);
  const unsigned int step = WARP_SIZE * gridDim.x;

  // final warp handles data outside of the warps_block partitions
  if (warp_id >= warps_block - 1) end = data_len;

  for (unsigned int pos = tix; pos < (bins_len + 1) * repeat; pos += bdx) Hs[pos] = 0;
  __syncthreads();

  for (unsigned int i = begin; i < end; i += step) {
    int d = in_data[i];
    // skip out-of-domain values (e.g. an incomp tile's raw fallback bits, not a quant code).
    if (d < 0 or d >= (int)bins_len) continue;
    atomicAdd(&Hs[off_rep + d], 1);
  }
  __syncthreads();

  for (unsigned int pos = tix; pos < bins_len; pos += bdx) {
    int sum = 0;
    for (int base = 0; base < (bins_len + 1) * repeat; base += bins_len + 1) {
      sum += Hs[base + pos];
    }
    atomicAdd(out_bins + pos, sum);
  }
}

}  // namespace psz

namespace psz::module {

template <typename T>
void GPU_histogram_generic<T>::init(
    size_t const data_len, uint16_t const hist_len, int& grid_dim, int& block_dim, int& shmem_use,
    int& r_per_block)
{
  int device_id, max_bytes, num_SMs;
  int items_per_thread;

  cudaGetDevice(&device_id);
  cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, device_id);

  //  query_maxbytes
  int max_bytes_opt_in;
  cudaDeviceGetAttribute(&max_bytes, cudaDevAttrMaxSharedMemoryPerBlock, device_id);

  // account for opt-in extra shared memory on certain architectures
  cudaDeviceGetAttribute(&max_bytes_opt_in, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
  max_bytes = std::max(max_bytes, max_bytes_opt_in);

  // config kernel attribute
  cudaFuncSetAttribute(
      (void*)KCU_p2013Histogram<T, uint32_t>,
      (cudaFuncAttribute)cudaFuncAttributeMaxDynamicSharedMemorySize, max_bytes);

  //  optimize_launch
  items_per_thread = 1;
  r_per_block = (max_bytes / sizeof(int)) / (hist_len + 1);
  grid_dim = num_SMs;
  // fits to size
  block_dim = ((((data_len / (grid_dim * items_per_thread)) + 1) / 64) + 1) * 64;
  while (block_dim > 1024) {
    if (r_per_block <= 1) { block_dim = 1024; }
    else {
      r_per_block /= 2;
      grid_dim *= 2;
      block_dim = ((((data_len / (grid_dim * items_per_thread)) + 1) / 64) + 1) * 64;
    }
  }
  shmem_use = ((hist_len + 1) * r_per_block) * sizeof(int);
}

template <typename T>
int GPU_histogram_generic<T>::kernel(
    T* in_data, size_t const data_len, uint32_t* out_hist, uint16_t const hist_len,
    int const grid_dim, int const block_dim, int const shmem_use, int const r_per_block,
    void* stream)
{
  KCU_p2013Histogram<<<grid_dim, block_dim, shmem_use, (cudaStream_t)stream>>>  //
      (in_data, data_len, out_hist, hist_len, r_per_block);

  return CUSZ_SUCCESS;
}

}  // namespace psz::module
