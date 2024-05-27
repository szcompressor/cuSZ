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

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>

#include "cusz/type.h"
#include "hf/hfstruct.h"
#include "hf/hfword.hh"
#include "mem/memseg_cxx.hh"

namespace cusz {

template <typename E, typename M = u4>
class HuffmanCodec {
 public:
  using BYTE = u1;
  using RAW = u1;
  using H4 = u4;
  using H8 = ull;

  static const int TYPICAL = sizeof(u4);
  static const int FAILSAFE = sizeof(u8);

 private:
  using BOOK4B = u4;
  using BOOK8B = u8;

  using SYM = E;

  // TODO psz and pszhf combined to use 128 byte
  struct alignas(128) pszhf_header {
    static const int HEADER = 0;
    static const int REVBK = 1;
    static const int PAR_NBIT = 2;
    static const int PAR_ENTRY = 3;
    static const int BITSTREAM = 4;
    static const int END = 5;

    // int self_bytes : 16;
    int bklen : 16;
    int sublen;
    int pardeg;
    size_t original_len;
    size_t total_nbit;
    size_t total_ncell;  // TODO change to uint32_t
    psz_dtype encdtype;
    M entry[END + 1];

    M compressed_size() const { return entry[END]; }
  };

  struct pszhf_rc {
    static const int SCRATCH = 0;
    static const int FREQ = 1;
    static const int BK = 2;
    static const int REVBK = 3;
    static const int PAR_NBIT = 4;
    static const int PAR_NCELL = 5;
    static const int PAR_ENTRY = 6;
    static const int BITSTREAM = 7;
    static const int END = 8;

    // uint32_t nbyte[END];
  };

  using RC = pszhf_rc;
  using pszhf_header = struct pszhf_header;
  using Header = pszhf_header;

  psz_dtype __encdtype;

 public:
  // array
  pszmem_cxx<RAW>* __scratch;
  pszmem_cxx<H4>* scratch4;
  pszmem_cxx<H8>* scratch8;

  pszmem_cxx<BYTE>* compressed;

  // pszmem_cxx<RAW>* __bk;
  pszmem_cxx<H4>* bk4;
  pszmem_cxx<H8>* bk8;

  pszmem_cxx<RAW>* __revbk;
  pszmem_cxx<BYTE>* revbk4;
  pszmem_cxx<BYTE>* revbk8;

  pszmem_cxx<RAW>* __bitstream;
  pszmem_cxx<H4>* bitstream4;
  pszmem_cxx<H8>* bitstream8;

  // data partition/embarrassingly parallelism description
  pszmem_cxx<M>* par_nbit;
  pszmem_cxx<M>* par_ncell;
  pszmem_cxx<M>* par_entry;

  MemU4* hist_view;

  // helper
  RC rc;
  // memory

  // timer
  float _time_book{0.0}, _time_lossless{0.0};

  hf_book* book_desc;
  hf_chunk* chunk_desc_d;
  hf_chunk* chunk_desc_h;
  hf_bitstream* bitstream_desc;

  int pardeg;
  int bklen;
  int numSMs;

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
  HuffmanCodec* decode(BYTE*, E*, uninit_stream_t, bool = true);
  HuffmanCodec* dump(std::vector<pszmem_dump>, char const*);
  HuffmanCodec* clear_buffer();

  // analysis
  void calculate_CR(
      MemU4* ectrl, szt sizeof_dtype = 4, szt overhead_bytes = 0);

 private:
  void __hf_merge(
      Header&, size_t const, int const, int const, int const,
      void* stream = nullptr);
  void hf_debug(const std::string, void*, int);

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
