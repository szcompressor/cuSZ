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
#include "utils/it_serial.hh"
#include "utils/timer.hh"

namespace psz::module {

template <typename E>
int SEQ_histogram_generic(
    E* in, size_t const inlen, uint32_t* out_hist, uint16_t const outlen, float* milliseconds)
{
  auto t1 = hires::now();
  for (auto i = 0; i < inlen; i++) {
    auto n = in[i];
    out_hist[(int)n] += 1;
  }
  auto t2 = hires::now();
  if (milliseconds) *milliseconds = static_cast<duration_t>(t2 - t1).count() * 1000;

  return CUSZ_SUCCESS;
}

}  // namespace psz::module

#define INIT_HIST_SEQ(E)                           \
  template int psz::module::SEQ_histogram_generic( \
      E* in, size_t const inlen, uint32_t* out_hist, uint16_t const outlen, float* milliseconds);

INIT_HIST_SEQ(u1);
INIT_HIST_SEQ(u2);
INIT_HIST_SEQ(u4);
INIT_HIST_SEQ(f4);

#undef INIT_HIST_SEQ
