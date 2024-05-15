/**
 * @file codec.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-23
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef DAB559E7_A5C1_4342_B17E_17C31DA96EEF
#define DAB559E7_A5C1_4342_B17E_17C31DA96EEF

#include <cuda.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>

#include "cusz/type.h"
#include "hfcxx_module.hh"
#include "hfword.hh"
#include "mem/memobj.hh"
#include "port.hh"
#include "utils/timer.hh"

namespace cusz {

using namespace portable;

template <typename E, typename M = u4, bool TIMING = true>
class HuffmanCodec {
 public:
  using BYTE = u1;
  using RAW = u1;
  using H4 = u4;
  using H = H4;

  static const int TYPICAL = sizeof(u4);
  static const int FAILSAFE = sizeof(u8);

  hires::time_point f, e, d, c, b, a;
  static const bool TimeBreakdown{false};

 private:
  using SYM = E;

  using phf_module = _2403::phf_kernel_wrapper<E, H, M, TIMING>;

  // TODO psz and pszhf combined to use 128 byte
  struct alignas(128) Header {
    static const int HEADER = 0;
    static const int REVBK = 1;
    static const int PAR_NBIT = 2;
    static const int PAR_ENTRY = 3;
    static const int BITSTREAM = 4;
    static const int END = 5;

    int bklen : 16;
    size_t sublen, pardeg;
    size_t original_len;
    size_t total_nbit, total_ncell;  // TODO change to uint32_t
    M entry[END + 1];

    M compressed_size() const { return entry[END]; }
  };

  Header header;

 public:
  // external
  MemU4* hist;

  struct internal_buffer;
  internal_buffer* buf;

  // timer
  float _time_book{0.0}, _time_lossless{0.0};

  size_t len;
  size_t pardeg, sublen;
  int numSMs;
  size_t bklen;

 public:
  ~HuffmanCodec();           // dtor
  HuffmanCodec() = default;  // ctor

  // getter
  float time_book() const;
  float time_lossless() const;

  // compile-time
  constexpr bool can_overlap_input_and_firstphase_encode();
  // public methods
  HuffmanCodec* init(
      size_t const, int const, int const, bool dbg_print = false);
  HuffmanCodec* build_codebook(
      uint32_t*, int const, uninit_stream_t = nullptr);

  HuffmanCodec* build_codebook(MemU4*, int const, uninit_stream_t = nullptr);

  HuffmanCodec* encode(E*, size_t const, BYTE**, size_t*, uninit_stream_t);

  HuffmanCodec* make_metadata();
  HuffmanCodec* decode(BYTE*, E*, uninit_stream_t, bool = true);
  HuffmanCodec* clear_buffer();

  // analysis
  void calculate_CR(
      memobj<E>* ectrl, MemU4* freq, szt sizeof_dtype = 4, szt overhead_bytes = 0);

 private:
  struct memcpy_helper {
    void* const ptr;
    size_t const nbyte;
    size_t const dst;
  };

  static int __revbk_bytes(int bklen, int BK_UNIT_BYTES, int SYM_BYTES)
  {
    static const int CELL_BITWIDTH = BK_UNIT_BYTES * 8;
    return BK_UNIT_BYTES * (2 * CELL_BITWIDTH) + SYM_BYTES * bklen;
  }

  static int revbk4_bytes(int bklen)
  {
    return __revbk_bytes(bklen, 4, sizeof(SYM));
  }
  static int revbk8_bytes(int bklen)
  {
    return __revbk_bytes(bklen, 8, sizeof(SYM));
  }
};

}  // namespace cusz

#endif /* DAB559E7_A5C1_4342_B17E_17C31DA96EEF */
