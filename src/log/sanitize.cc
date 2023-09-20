/**
 * @file sanitize.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.9
 * @date 2023-09-07
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include "log/sanitize.hh"

#include "busyheader.hh"
#include "context.h"
#include "hf/hfword.hh"

// resemble the one defined in dbg_cu.inl
#define __PSZSANITIZE_VAR(LOC, VAR)       \
  if (string(LOC) != "")                  \
    __PSZLOG__STATUS_SANITIZE_IN(LOC)     \
  else                                    \
    __PSZLOG__STATUS_SANITIZE             \
  cout << "(var) " << #VAR << "=" << VAR; \
  __PSZLOG__NEWLINE

template <typename T, typename E, typename H>
void psz::sanitize<T, E, H>::sanitize_pszctx(
    pszctx const* const ctx, std::string LOC)
{
  __PSZSANITIZE_VAR(LOC.c_str(), ctx->radius)
  __PSZSANITIZE_VAR(LOC.c_str(), ctx->dict_size)
}

template <typename T, typename E, typename H>
void psz::sanitize<T, E, H>::sanitize_quantcode(
    E const* h_ectrl, szt len, szt bklen)
{
  if (not(std::is_same<E, u1>::value or std::is_same<E, u2>::value or
          std::is_same<E, u4>::value))
    __PSZDBG__FATAL("E is not valid dtype for histogram input.");

  auto found_ptr = std::find_if(
      h_ectrl, h_ectrl + len, [&](auto v) { return v >= bklen or v < 0; });
  if (found_ptr != h_ectrl + len) {
    auto idx = found_ptr - h_ectrl;

    __PSZDBG__FATAL(
        "The first found invalid point (value >= bklen) is quantcode[" +
        to_string(idx) + "] = " + to_string(h_ectrl[idx]))
  }
  else {
    __PSZLOG__STATUS_SANITIZE
    __PSZLOG__P("Quantization codes (the hist input) are all valid.")
  }
}

template <typename T, typename E, typename H>
void psz::sanitize<T, E, H>::sanitize_hist_book(
    M const* h_hist, H const* h_bk, szt bklen)
{
  using PW = PackedWordByWidth<sizeof(H)>;

  cout << "[psz::dbg::(hist,bk)] printing non-zero frequencies" << endl;
  for (auto i = 0; i < bklen; i++) {
    auto freq = h_hist[i];
    auto _ = h_bk[i];
    auto packed_word = reinterpret_cast<PW*>(&_);
    if (freq != 0)
      printf(
          "\e[90m[psz::dbg::(hist,bk)]\e[0m "
          "idx=%4d\tfreq=%u\t",
          i, freq);
    cout << "packed(bits,word)\t"
         << std::bitset<PW::FIELDWIDTH_bits>(packed_word->bits) << "  ";
    cout << std::bitset<PW::FIELDWIDTH_bits>(packed_word->word) << endl;
  }
}

template <typename T, typename E, typename H>
void psz::sanitize<T, E, H>::sanitize_hist_out(
    M const* const h_hist, szt bklen)
{
  if (not(std::is_same<M, u1>::value or std::is_same<M, u2>::value or
          std::is_same<M, u4>::value))
    __PSZDBG__FATAL("M is not valid dtype for histogram output.");

  // TODO warning
  auto all_zero =
      std::all_of(h_hist, h_hist + bklen, [](auto i) { return i == 0; });
  if (all_zero) __PSZDBG__FATAL("[psz::error] histogram outputs all zeros");

  cout << "[psz::dbg::hist::out] printing non-zero frequencies" << endl;
  std::for_each(
      h_hist, h_hist + bklen, [quantcode = 0, idx2 = 0](auto freq) mutable {
        if (freq != 0)
          printf(
              "\e[90m[psz::dbg::hist_out]\e[0m quantcode=%4d (nonzero %4d-th)\tfreq=%u\n",
              quantcode, idx2++, freq);
        quantcode++;
      });
}

template struct psz::sanitize<f4, u4, u4>;
template struct psz::sanitize<f8, u4, u4>;
template struct psz::sanitize<f4, u4, u8>;
template struct psz::sanitize<f8, u4, u8>;
template struct psz::sanitize<f4, u4, ull>;
template struct psz::sanitize<f8, u4, ull>;
