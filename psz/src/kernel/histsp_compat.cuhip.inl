/**
 * @file histsp.cuhip.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-05-18
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <cstdint>

#include "detail/histsp.cuhip.inl"
#include "module/cxx_module.hh"
#include "utils/timer.hh"

namespace psz::cuhip {

template <typename T, typename FQ>
int GPU_histogram_sparse(
    T* in, uint32_t inlen, FQ* out_hist, uint32_t outlen, float* milliseconds,
    cudaStream_t stream)
{
  auto chunk = 32768;
  auto num_chunks = (inlen - 1) / chunk + 1;
  auto num_workers = 256;  // n SIMD-32

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  psz::KERNEL_CUHIP_histogram_sparse_multiwarp<T, FQ>
      <<<num_chunks, num_workers, sizeof(FQ) * outlen, stream>>>(
          in, inlen, chunk, out_hist, outlen, outlen / 2);
  STOP_GPUEVENT_RECORDING(stream);

  cudaStreamSynchronize(stream);
  TIME_ELAPSED_GPUEVENT(milliseconds);
  DESTROY_GPUEVENT_PAIR;

  return 0;
}

}  // namespace psz::cuhip

////////////////////////////////////////////////////////////////////////////////
#define SPECIALIZE_PSZCXX_COMPAT_MODULE_HIST_CAUCHY(BACKEND, E)           \
  template <>                                                             \
  int pszcxx_compat_histogram_cauchy<BACKEND, E, uint32_t>(               \
      E * in, uint32_t inlen, uint32_t * out_hist, uint32_t outlen,       \
      float* milliseconds, void* stream)                                  \
  {                                                                       \
    return psz::cuhip::GPU_histogram_sparse<E, uint32_t>(                 \
        in, inlen, out_hist, outlen, milliseconds, (cudaStream_t)stream); \
  }
