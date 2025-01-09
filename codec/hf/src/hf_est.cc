#include "hf_est.h"

#include <numeric>
#include <stdexcept>

#include "hfbk_impl.hh"
#include "hfword.hh"
#include "utils/vis_stat.hh"

namespace {

template <int HF_SYM_BYTE = 4>
void est_impl(uint32_t* freq, int const bklen, double* entropy, double* cr)
{
  using H = std::conditional_t<HF_SYM_BYTE == 4, uint32_t, uint64_t>;
  auto book = new H[bklen];
  phf_CPU_build_codebook_v2<H>(freq, bklen, book);

  auto all_bits = 0u;
  auto len = 0u;

  len = std::accumulate(freq, freq + bklen, 0);

  for (auto i = 0; i < bklen; i++) {
    auto count = freq[i];
    if (count != 0) {
      using PW = HuffmanWord<HF_SYM_BYTE>;
      auto pw = (PW*)(&book[i]);
      auto bits = pw->bitcount;
      all_bits += bits * count;

      auto p = freq[i] * 1.0 / len;
      *entropy += -std::log2(p) * p;

      // printf(
      //     "%d:\t%d\tbits:\t%d\t\tpartial:\t%d\n", i, count, (int)bits,
      //     (int)bits * count);
    }
  }

  auto dtype_width = sizeof(decltype(freq[0])) * 8;
  *cr = (dtype_width * len * 1.0 / all_bits * 1.0);

  // printf(
  //     "entropy:\t%.3f\t(upper limit: %.3f)\tCR:\t%.3f\n", *entropy,
  //     (32 / *entropy), *cr);

  delete[] book;
}

}  // namespace

void pszanalysis_hf_buildtree(
    uint32_t* freq, int const bklen, double* entropy, double* cr, int const symbol_byte)
{
  if (symbol_byte == 4)
    est_impl<4>(freq, bklen, entropy, cr);
  else if (symbol_byte == 8)
    est_impl<8>(freq, bklen, entropy, cr);
  else
    throw std::runtime_error("4 or 8; otherwise not working for this fixed-tree estimation.");
}