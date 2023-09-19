/**
 * @file histsp.seq.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-07-26
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <cstdint>

#include "kernel/histsp.hh"
#include "utils/timer.hh"

namespace psz {
namespace detail {

// temporarily, there should be no obvious speed up than the normal hist on
// CPU.
template <typename T, typename FQ>
int histsp_cpu_v1(T* in, uint32_t inlen, FQ* out_hist, uint32_t outlen)
{
  auto radius = outlen / 2;
  T center{0}, neg1{0}, pos1{0};

  for (auto i = 0; i < inlen; i++) {
    auto n = in[i];
    if (n == radius)
      center++;
    else if (n == radius - 1)
      neg1++;
    else if (n == radius + 1)
      pos1++;
    else
      out_hist[n]++;
  }
  out_hist[radius] = center;
  out_hist[radius - 1] = neg1;
  out_hist[radius + 1] = pos1;

  return 0;
}

template <typename T, typename FQ>
int histsp_cpu_v2(
    T* in, uint32_t inlen, FQ* out_hist, uint32_t outlen, float* milliseconds)
{
  auto radius = outlen / 2;
  T neg1{0}, pos1{0};

  auto start = hires::now();
  {
    for (auto i = 0; i < inlen; i++) {
      auto n = in[i];
      if (n == radius)
        continue;
      else if (n == radius - 1)
        neg1++;
      else if (n == radius + 1)
        pos1++;
      else
        out_hist[(int)n]++;
    }
    out_hist[radius - 1] = neg1;
    out_hist[radius + 1] = pos1;

    auto sum = 0U;
    for (auto i = 0; i < outlen; i++) sum += out_hist[i];
    out_hist[radius] = inlen - sum;
  }
  auto end = hires::now();

  *milliseconds = static_cast<duration_t>(end - start).count() * 1000;

  return 0;
}

}  // namespace detail
}  // namespace psz

#define SPECIALIZE_CPU(E)                                           \
  template <>                                                       \
  int psz::histsp<pszpolicy::SEQ, E, uint32_t>(                         \
      E * in, uint32_t inlen, uint32_t * out_hist, uint32_t outlen, \
      float* milliseconds, void* stream)                            \
  {                                                                 \
    return psz::detail::histsp_cpu_v2<E, uint32_t>(                 \
        in, inlen, out_hist, outlen, milliseconds);                 \
  }

SPECIALIZE_CPU(uint8_t)
SPECIALIZE_CPU(uint16_t)
SPECIALIZE_CPU(uint32_t)
SPECIALIZE_CPU(float)

#undef SPECIALIZE_CPU