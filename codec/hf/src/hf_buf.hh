/**
 * @file hf_buf.hh (PRIVATE)
 * @brief Private full definition of phf::Buf buffer management class
 */

#ifndef HF_BUF_HH_PRIVATE
#define HF_BUF_HH_PRIVATE

#include <memory>

#include "c_type.h"
#include "hf.h"

namespace phf {

/**
 * @brief Buffer management for Huffman encoding/decoding (PRIVATE IMPLEMENTATION)
 *
 * This class manages GPU and host memory for Huffman compression operations.
 * It uses the pimpl (pointer-to-implementation) pattern to hide implementation details.
 *
 * Users should only interact with this via the public phf::high_level API.
 */
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
  using H4 = u4;
  using M = PHF_METADATA;
  using Header = phf_header;

  // constructor/destructor
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

  // HFR breaking-point sparse buffers
  E* brval_d() const;
  u4* bridx_d() const;
  u4* brnum_d() const;

  void update_header(phf_header& header);
  void calc_offset(phf_header& header, M* byte_offsets);

  // other methods
  void memcpy_merge(phf_header& header, phf_stream_t stream);
  void clear_buffer();
};

}  // namespace phf

#endif /* HF_BUF_HH_PRIVATE */
