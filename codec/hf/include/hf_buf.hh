#ifndef PHF_HF_BUF
#define PHF_HF_BUF

#include <cstdint>

#include "hf_type.h"
#include "mem/cxx_backends.h"

namespace phf {

template <typename E>
struct Buf {
  using H4 = u4;
  using M = PHF_METADATA;

  // helper
  typedef struct RC {
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
  } RC;

  typedef struct {
    void* const ptr;
    size_t const nbyte;
    size_t const dst;
  } memcpy_helper;

  using SYM = E;
  using Header = phf_header;

  // vars
  const size_t len;
  size_t pardeg;
  size_t sublen;
  const size_t bklen;
  const bool use_HFR;
  const size_t revbk4_bytes;
  const size_t bitstream_max_len;

  u2 rt_bklen;
  int numSMs;

  // array
  GPU_unique_dptr<H4[]> d_scratch4;
  GPU_unique_hptr<H4[]> h_scratch4;
  PHF_BYTE* d_encoded;
  PHF_BYTE* h_encoded;
  GPU_unique_dptr<H4[]> d_bitstream4;
  GPU_unique_hptr<H4[]> h_bitstream4;

  GPU_unique_dptr<H4[]> d_bk4;
  GPU_unique_hptr<H4[]> h_bk4;
  GPU_unique_dptr<PHF_BYTE[]> d_revbk4;
  GPU_unique_hptr<PHF_BYTE[]> h_revbk4;

  // data partition/embarrassingly parallelism description
  GPU_unique_dptr<M[]> d_par_nbit;
  GPU_unique_hptr<M[]> h_par_nbit;
  GPU_unique_dptr<M[]> d_par_ncell;
  GPU_unique_hptr<M[]> h_par_ncell;
  GPU_unique_dptr<M[]> d_par_entry;
  GPU_unique_hptr<M[]> h_par_entry;

  // ReVISIT-lite specific
  GPU_unique_dptr<E[]> d_brval;
  GPU_unique_dptr<u4[]> d_bridx;
  GPU_unique_dptr<u4[]> d_brnum;
  GPU_unique_hptr<u4[]> h_brnum;

  // utils
  static int _revbk4_bytes(int bklen);
  static int _revbk8_bytes(int bklen);

  // ctor
  Buf(size_t inlen, size_t _bklen, int _pardeg = -1, bool _use_HFR = false, bool debug = false);
  ~Buf();

  // setter
  void register_runtime_bklen(int _rt_bklen) { rt_bklen = _rt_bklen; }

  void memcpy_merge(phf_header& header, phf_stream_t stream);
  void clear_buffer();
};

}  // namespace phf

#endif /* PHF_HF_BUF */
