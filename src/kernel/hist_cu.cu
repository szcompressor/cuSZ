/**
 * @file stat_g.cu
 * @author Cody Rivera, Jiannan Tian
 * @brief Fast histogramming from [Gómez-Luna et al. 2013], wrapper
 * @version 0.3
 * @date 2022-11-02
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <cstdint>

#include "../kernel/detail/hist_cu.inl"
#include "cusz/type.h"
#include "kernel/hist.hh"

namespace psz {
namespace detail {

template <typename T>
cusz_error_status histogram_cuda(
    T* in, size_t const inlen, uint32_t* out_hist, int const outlen,
    float* milliseconds, cudaStream_t stream)
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
        kernel::p2013Histogram<T, uint32_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_bytes);
  };

  auto optimize_launch = [&]() {
    items_per_thread = 1;
    r_per_block = (max_bytes / sizeof(int)) / (outlen + 1);
    grid_dim = num_SMs;
    // fits to size
    block_dim =
        ((((inlen / (grid_dim * items_per_thread)) + 1) / 64) + 1) * 64;
    while (block_dim > 1024) {
      if (r_per_block <= 1) { block_dim = 1024; }
      else {
        r_per_block /= 2;
        grid_dim *= 2;
        block_dim =
            ((((inlen / (grid_dim * items_per_thread)) + 1) / 64) + 1) * 64;
      }
    }
    shmem_use = ((outlen + 1) * r_per_block) * sizeof(int);
  };

  query_maxbytes();
  optimize_launch();

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  kernel::p2013Histogram<<<grid_dim, block_dim, shmem_use, stream>>>  //
      (in, out_hist, inlen, outlen, r_per_block);

  STOP_GPUEVENT_RECORDING(stream);

  cudaStreamSynchronize(stream);
  TIME_ELAPSED_GPUEVENT(milliseconds);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}

}  // namespace detail
}  // namespace psz

#define SPECIALIZE_HIST_CUDA(T)                                         \
  template <>                                                           \
  cusz_error_status psz::histogram<pszpolicy::CUDA, T>(          \
      T * in, size_t const inlen, uint32_t* out_hist, int const nbin,   \
      float* milliseconds, void* stream)                                \
  {                                                                     \
    return psz::detail::histogram_cuda<T>(                              \
        in, inlen, out_hist, nbin, milliseconds, (cudaStream_t)stream); \
  }

SPECIALIZE_HIST_CUDA(uint8_t);
SPECIALIZE_HIST_CUDA(uint16_t);
SPECIALIZE_HIST_CUDA(uint32_t);
SPECIALIZE_HIST_CUDA(float);
// SPECIALIZE_HIST_CUDA(uint64_t);

#undef SPECIALIZE_HIST_CUDA
