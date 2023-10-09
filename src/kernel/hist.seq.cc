/**
 * @file hist.seq.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-07-26
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include "kernel/hist.hh"
#include "utils/timer.hh"
#include "utils/it_serial.hh"

namespace psz {
namespace detail {

template <typename T>
psz_error_status histogram_seq(
    T* in, size_t const inlen, uint32_t* out_hist, int const outlen,
    float* milliseconds)
{
  auto t1 = hires::now();
  for (auto i = 0; i < inlen; i++) {
    auto n = in[i];
    out_hist[(int)n] += 1;
  }
  auto t2 = hires::now();
  *milliseconds = static_cast<duration_t>(t2 - t1).count() * 1000;

  return CUSZ_SUCCESS;
}

}  // namespace detail
}  // namespace psz

#define SPECIALIZE_HIST_SER(T)                                        \
  template <>                                                         \
  psz_error_status psz::histogram<pszpolicy::SEQ, T>(         \
      T * in, size_t const inlen, uint32_t* out_hist, int const nbin, \
      float* milliseconds, void* stream)                              \
  {                                                                   \
    return psz::detail::histogram_seq<T>(                             \
        in, inlen, out_hist, nbin, milliseconds);                     \
  }

SPECIALIZE_HIST_SER(uint8_t);
SPECIALIZE_HIST_SER(uint16_t);
SPECIALIZE_HIST_SER(uint32_t);
SPECIALIZE_HIST_SER(float);
// SPECIALIZE_HIST_CUDA(uint64_t);

#undef SPECIALIZE_HIST_SER
