/**
 * @file histsp.hip
 * @author Jiannan Tian
 * @brief For HIP backend
 * @version 0.4
 * @date 2023-08-20
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <hip/hip_runtime.h>

#include <cstdint>

#include "detail/histsp_cu.inl"
#include "kernel/histsp.hh"
#include "utils/timer.hh"

namespace psz {
namespace detail {

template <typename T, typename FQ>
int histsp_hip(
    T* in, uint32_t inlen, FQ* out_hist, uint32_t outlen, float* milliseconds,
    hipStream_t stream)
{
  auto chunk = 32768;
  auto num_chunks = (inlen - 1) / chunk + 1;
  auto num_workers = 256;  // n SIMD-32 (or 64 if AMD)

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  histsp_multiwarp<T, FQ>
      <<<num_chunks, num_workers, sizeof(FQ) * outlen, stream>>>(
          in, inlen, chunk, out_hist, outlen, outlen / 2);
  STOP_GPUEVENT_RECORDING(stream);

  hipStreamSynchronize(stream);
  TIME_ELAPSED_GPUEVENT(milliseconds);
  DESTROY_GPUEVENT_PAIR;

  return 0;
}

}  // namespace detail
}  // namespace psz

#define SPECIALIZE_HIP(E)                                                \
  template <>                                                            \
  int psz::histsp<pszpolicy::HIP, E, uint32_t>(                          \
      E * in, uint32_t inlen, uint32_t * out_hist, uint32_t outlen,      \
      float* milliseconds, void* stream)                                 \
  {                                                                      \
    return psz::detail::histsp_hip<E, uint32_t>(                         \
        in, inlen, out_hist, outlen, milliseconds, (hipStream_t)stream); \
  }

SPECIALIZE_HIP(uint8_t)
SPECIALIZE_HIP(uint16_t)
SPECIALIZE_HIP(uint32_t)
SPECIALIZE_HIP(float)

#undef SPECIALIZE_HIP
