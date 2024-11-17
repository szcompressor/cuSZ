/**
 * @file hist.cuhip.inl
 * @author Cody Rivera (cjrivera1@crimson.ua.edu), Megan Hickman Fulp
 * (mlhickm@g.clemson.edu)
 * @brief Fast histogramming from [Gómez-Luna et al. 2013]
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-02-16
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory See LICENSE in top-level directory
 *
 */

#ifndef D69BE972_2A8C_472E_930F_FFAB041F3F2B
#define D69BE972_2A8C_472E_930F_FFAB041F3F2B

#include <cstdio>
#include <limits>

#include "typing.hh"
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

template <typename T, typename FREQ>
__global__ void KERNEL_CUHIP_histogram_naive(
    T* in_data, size_t const data_len, FREQ* out_bins, uint16_t const bins_len,
    uint16_t const repeat)
{
  auto i = blockDim.x * blockIdx.x + threadIdx.x;
  auto j = 0u;
  if (i * repeat < data_len) {  // if there is a symbol to count,
    for (j = i * repeat; j < (i + 1) * repeat; j++) {
      if (j < data_len) {
        auto item = in_data[j];         // symbol to count
        atomicAdd(&out_bins[item], 1);  // update bin count by 1
      }
    }
  }
}

/* Copied from J. Gomez-Luna et al. */
template <typename T, typename FREQ>
__global__ void KERNEL_CUHIP_p2013Histogram(
    T* in_data, size_t const data_len, FREQ* out_bins, uint16_t const bins_len,
    uint16_t const repeat)
{
  extern __shared__ int Hs[/*(bins_len + 1) * repeat*/];

  const unsigned int warp_id = (int)(tix / WARP_SIZE);
  const unsigned int lane = tix % WARP_SIZE;
  const unsigned int warps_block = bdx / WARP_SIZE;
  const unsigned int off_rep = (bins_len + 1) * (tix % repeat);
  const unsigned int begin =
      (data_len / warps_block) * warp_id + WARP_SIZE * blockIdx.x + lane;
  unsigned int end = (data_len / warps_block) * (warp_id + 1);
  const unsigned int step = WARP_SIZE * gridDim.x;

  // final warp handles data outside of the warps_block partitions
  if (warp_id >= warps_block - 1) end = data_len;

  for (unsigned int pos = tix; pos < (bins_len + 1) * repeat; pos += bdx)
    Hs[pos] = 0;
  __syncthreads();

  for (unsigned int i = begin; i < end; i += step) {
    int d = in_data[i];
    d = d <= 0 and d >= bins_len ? bins_len / 2 : d;
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

namespace psz::cuhip {

template <typename T>
psz_error_status GPU_histogram_generic(
    T* in_data, size_t const data_len, uint32_t* out_hist,
    uint16_t const hist_len, float* milliseconds, cudaStream_t stream)
{
  int device_id, max_bytes, num_SMs;
  int items_per_thread, r_per_block, grid_dim, block_dim, shmem_use;

  cudaGetDevice(&device_id);
  cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, device_id);

  auto query_maxbytes = [&]() {
    int max_bytes_opt_in;
    cudaDeviceGetAttribute(
        &max_bytes, cudaDevAttrMaxSharedMemoryPerBlock, device_id);

    // account for opt-in extra shared memory on certain architectures
    cudaDeviceGetAttribute(
        &max_bytes_opt_in, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
    max_bytes = std::max(max_bytes, max_bytes_opt_in);

    // config kernel attribute
    cudaFuncSetAttribute(
        (void*)KERNEL_CUHIP_p2013Histogram<T, uint32_t>,
        (cudaFuncAttribute)cudaFuncAttributeMaxDynamicSharedMemorySize,
        max_bytes);
  };

  auto optimize_launch = [&]() {
    items_per_thread = 1;
    r_per_block = (max_bytes / sizeof(int)) / (hist_len + 1);
    grid_dim = num_SMs;
    // fits to size
    block_dim =
        ((((data_len / (grid_dim * items_per_thread)) + 1) / 64) + 1) * 64;
    while (block_dim > 1024) {
      if (r_per_block <= 1) { block_dim = 1024; }
      else {
        r_per_block /= 2;
        grid_dim *= 2;
        block_dim =
            ((((data_len / (grid_dim * items_per_thread)) + 1) / 64) + 1) * 64;
      }
    }
    shmem_use = ((hist_len + 1) * r_per_block) * sizeof(int);
  };

  query_maxbytes();
  optimize_launch();

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  KERNEL_CUHIP_p2013Histogram<<<grid_dim, block_dim, shmem_use, stream>>>  //
      (in_data, data_len, out_hist, hist_len, r_per_block);

  STOP_GPUEVENT_RECORDING(stream);

  cudaStreamSynchronize(stream);
  TIME_ELAPSED_GPUEVENT(milliseconds);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}

}  // namespace psz::cuhip

#endif /* D69BE972_2A8C_472E_930F_FFAB041F3F2B */
