/**
 * @file hist_sp.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-05-18
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <cstdint>

#include "detail/histsp.inl"
#include "kernel/histsp.hh"

namespace psz {
namespace detail {

template <typename T, typename FQ>
int histsp_cuda(
    T* in, uint32_t inlen, FQ* out_hist, uint32_t outlen, cudaStream_t stream)
{
  constexpr auto CHUNK = 32768;
  constexpr auto NWARP = 8;
  constexpr auto NTREAD = 32 * NWARP;

  histsp_multiwarp<T, NWARP, CHUNK, FQ>
      <<<(inlen - 1) / CHUNK + 1, NTREAD, sizeof(FQ) * outlen, stream>>>(
          in, inlen, out_hist, outlen, outlen / 2);
  cudaStreamSynchronize(stream);

  return 0;
}

}  // namespace detail
}  // namespace psz

#define SPECIALIZE_CUDA(E)                                          \
  template <>                                                       \
  int histsp<psz_policy::CUDA, E, uint32_t>(                        \
      E * in, uint32_t inlen, uint32_t * out_hist, uint32_t outlen, \
      void* stream)                                                 \
  {                                                                 \
    return psz::detail::histsp_cuda<E, uint32_t>(                   \
        in, inlen, out_hist, outlen, (cudaStream_t)stream);         \
  }

SPECIALIZE_CUDA(uint8_t)
SPECIALIZE_CUDA(uint16_t)
SPECIALIZE_CUDA(uint32_t)
SPECIALIZE_CUDA(float)

#undef SPECIALIZE_CUDA
