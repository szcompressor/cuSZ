/**
 * @file hf_hl.hh
 * @brief High-level Huffman encoding/decoding interface with runtime buffer definition
 */

#ifndef HF_HL_HH
#define HF_HL_HH

#include <memory>

#include "c_type.h"
#include "hf.h"

namespace phf {

// namespace-wide type aliasing
using H4 = u4;
using M = PHF_METADATA;
using HF_STREAM = void*;

template <typename E>
struct Buf {
  struct impl;
  std::unique_ptr<impl> pimpl;

  // helper
  typedef struct RC {
    static const int SCRATCH = 0;
    static const int FREQ = 1;
    static const int BK = 2;
    static const int RVBK = 3;
    static const int PAR_NBIT = 4;
    static const int PAR_NCELL = 5;
    static const int PAR_ENTRY = 6;
    static const int BITSTREAM = 7;
    static const int END = 8;
  } RC;

  typedef struct {
    void* const ptr;
    size_t const nbyte;
    size_t const dst;
  } memcpy_helper;

  using SYM = E;
  using Header = phf_header;

  // ctor
  Buf(size_t inlen, size_t _bklen, int _pardeg = -1, bool _use_HFR = false, bool debug = false);
  ~Buf();

  // setter
  void register_runtime_bklen(int const rt_bklen);

  // getter: variables
  u2 rt_bklen() const;
  int numSMs() const;
  size_t sublen() const;
  size_t pardeg() const;
  size_t bitstream_max_len() const;
  size_t rvbk_bytes() const;

  // getter: arrays
  H4* book_d() const;
  H4* book_h() const;
  u1* rvbk_d() const;
  u1* rvbk_h() const;
  H4* scratch_d() const;
  H4* scratch_h() const;
  M* par_nbit_d() const;
  M* par_nbit_h() const;
  M* par_ncell_d() const;
  M* par_ncell_h() const;
  M* par_entry_d() const;
  M* par_entry_h() const;
  H4* bitstream_d() const;
  H4* bitstream_h() const;
  PHF_BYTE* encoded_d() const;
  PHF_BYTE* encoded_h() const;

  void update_header(phf_header& header);
  void calc_offset(phf_header& header, M* byte_offsets);

  // other methods
  void memcpy_merge(phf_header& header, phf_stream_t stream);
  void clear_buffer();
};

#define PHF_BUF phf::Buf<E>
#define PHF_STREAM void*

template <typename E>
struct high_level {
  static int build_book(PHF_BUF* buf, u4* h_hist, u2 const runtime_bklen, PHF_STREAM stream);

  static int encode(
      PHF_BUF* buf, E* in_data, size_t const data_len, uint8_t** out_encoded, size_t* encoded_len,
      phf_header& header, PHF_STREAM stream);

  static int encode_ReVISIT_lite(
      PHF_BUF* buf, E* in_data, size_t const data_len, uint8_t** out_encoded, size_t* encoded_len,
      phf_header& header, PHF_STREAM stream);

  static int decode(
      PHF_BUF* buf, phf_header& header, PHF_BYTE* in_encoded, E* out_decoded, PHF_STREAM stream);
};

}  // namespace phf

#undef PHF_BUF
#undef PHF_STREAM

#endif /* HF_HL_HH */
