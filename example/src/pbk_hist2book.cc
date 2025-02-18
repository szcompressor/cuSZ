#include <cstdlib>
#include <iostream>
#include <string>

#include "cxx_hfbk.h"
#include "hfclass.hh"
#include "hfword.hh"
#include "mem/cxx_backends.h"
#include "utils/io.hh"

using _portable::utils::fromfile;
using _portable::utils::tofile;

static constexpr auto PBK_LEN = 128;
static constexpr auto PBK_N = 11;

using Freq = u4;
using Hf = u4;

const char* ifname_pbk_hist;
u4 PBK_REVBK_BYTES;

#define OFNAME(OLD_INFIX, NEW_INFIX) ofname(ifname_pbk_hist, OLD_INFIX, NEW_INFIX)

std::string ofname(
    std::string const hist_name, std::string const from = "_hist", std::string const to = "_revbk")
{
  std::string oname(hist_name);

  size_t pos = oname.find(from);
  if (pos != std::string::npos) oname.replace(pos, from.length(), to);

  return oname;
}

template <typename E>
void build_books()
{
  ifname_pbk_hist = std::getenv("PBK_HIST");

  if (not ifname_pbk_hist) {
    std::cout << "ENV VAR PBK_HIST is not set." << std::endl;
    exit(1);
  }
  else {
    auto PBK_REVBK_BYTES = phf_reverse_book_bytes(PBK_LEN, 4, sizeof(E));

    auto h_pbk_hist = MAKE_UNIQUE_HOST(Freq, PBK_LEN * PBK_N);
    auto h_pbk_r64 = MAKE_UNIQUE_HOST(Hf, PBK_LEN * PBK_N);
    auto h_pbk_revbk_r64 = MAKE_UNIQUE_HOST(u1, PBK_REVBK_BYTES * PBK_N);

    fromfile(ifname_pbk_hist, h_pbk_hist.get(), PBK_LEN * PBK_N);

    for (auto i = 0; i < PBK_N; i++) {
      auto this_hist = h_pbk_hist.get() + i * PBK_LEN;
      auto this_book = h_pbk_r64.get() + i * PBK_LEN;
      auto this_rvbk = h_pbk_revbk_r64.get() + i * PBK_REVBK_BYTES;
      phf_CPU_build_canonized_codebook_v2<E, Hf>(
          this_hist, PBK_LEN, this_book, this_rvbk, PBK_REVBK_BYTES);

      if constexpr (0) {
        printf("(pbk_hist2book) tree id: %d\n", i);
        phf_revbook_view<u2, u4>(
            h_pbk_revbk_r64.get() + i * PBK_REVBK_BYTES, PBK_LEN, true, true, false);
        printf("----------------------------------------\n");
      }
    }

    auto ofname_book = OFNAME("_hist_u4", "_book_u4");
    auto ofname_rvbk = OFNAME("_hist_u4", "_rvbk_u1");

    std::cout << "(input)  PBK_HIST: " << ifname_pbk_hist << std::endl;
    // std::cout << "(output) PBK_BOOK: " << ofname_book << std::endl;
    std::cout << "(output) PBK_RVBK: " << ofname_rvbk << std::endl;

    // tofile(ofname_book.c_str(), h_pbk_r64.get(), PBK_LEN * PBK_N);
    tofile(ofname_rvbk.c_str(), h_pbk_revbk_r64.get(), PBK_REVBK_BYTES * PBK_N);

    if constexpr (1) {
      auto check_rvbk = MAKE_UNIQUE_HOST(u1, PBK_REVBK_BYTES * PBK_N);
      fromfile(ofname_rvbk.c_str(), check_rvbk.get(), PBK_REVBK_BYTES * PBK_N);

      for (auto i = 0; i < PBK_N; i++) {
        printf("(check rvbk) tree id: %d\n", i);
        phf_revbook_view<u2, u4>(
            h_pbk_revbk_r64.get() + i * PBK_REVBK_BYTES, PBK_LEN, true, true, false);
        printf("----------------------------------------\n");
      }
    }
  }
}

int main()
{
  build_books<u2>();  // u2 is the current default
  return 0;
}