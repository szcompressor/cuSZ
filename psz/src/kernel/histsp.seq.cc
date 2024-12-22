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

#include "module/cxx_module.hh"
#include "utils/timer.hh"

namespace psz::module {

// There should be no obvious speed up than the normal hist on CPU.
template <typename E>
int SEQ_histogram_Cauchy_v1(
    E* in, size_t const inlen, uint32_t* out_hist, uint16_t const outlen)
{
  auto radius = outlen / 2;
  E center{0}, neg1{0}, pos1{0};

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

template <typename E>
int SEQ_histogram_Cauchy_v2(
    E* in, size_t const inlen, uint32_t* out_hist, uint16_t const outlen,
    float* milliseconds)
{
  auto radius = outlen / 2;
  E neg1{0}, pos1{0};

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

}  // namespace psz::module

#define INIT_HISTSP_SEQ(E)                                                   \
  template int psz::module::SEQ_histogram_Cauchy_v2<E>(                      \
      E * in, size_t const inlen, uint32_t* out_hist, uint16_t const outlen, \
      float* milliseconds);

INIT_HISTSP_SEQ(u1)
INIT_HISTSP_SEQ(u2)
INIT_HISTSP_SEQ(u4)
INIT_HISTSP_SEQ(f4)

#undef INIT_HISTSP_SEQ