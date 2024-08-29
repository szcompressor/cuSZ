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

namespace psz {

// temporarily, there should be no obvious speed up than the normal hist on
// CPU.
template <typename T, typename FQ>
int KERNEL_SEQ_histogram_sparse_v1(
    T* in, uint32_t inlen, FQ* out_hist, uint32_t outlen)
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
int KERNEL_SEQ_histogram_sparse_v2(
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

}  // namespace psz

#define SPECIALIZE_PSZCXX_COMPAT_MODULE_HIST_CAUCHY(E)              \
  template <>                                                       \
  int pszcxx_compat_histogram_cauchy<psz_policy::SEQ, E, uint32_t>( \
      E * in, uint32_t inlen, uint32_t * out_hist, uint32_t outlen, \
      float* milliseconds, void* stream)                            \
  {                                                                 \
    return psz::KERNEL_SEQ_histogram_sparse_v2<E, uint32_t>(        \
        in, inlen, out_hist, outlen, milliseconds);                 \
  }

SPECIALIZE_PSZCXX_COMPAT_MODULE_HIST_CAUCHY(u1)
SPECIALIZE_PSZCXX_COMPAT_MODULE_HIST_CAUCHY(u2)
SPECIALIZE_PSZCXX_COMPAT_MODULE_HIST_CAUCHY(u4)
SPECIALIZE_PSZCXX_COMPAT_MODULE_HIST_CAUCHY(f4)

#undef SPECIALIZE_PSZCXX_COMPAT_MODULE_HIST_CAUCHY