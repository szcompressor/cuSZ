#ifndef HF_HL_HH
#define HF_HL_HH

#include <cstdint>
#include <memory>

#include "c_type.h"
#include "hf.h"
#include "hf_impl.hh"
#include "mem/cxx_backends.h"
#include "mem/cxx_sp_gpu.h"
#include "utils/io.hh"
#include "utils/timer.hh"

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

namespace phf {

template <typename E>
class [[deprecated("Moving to non-OOD Huffman encoding.")]] HuffmanCodec {
 private:
  using SYM = E;
  using Buf = phf::Buf<E>;

 public:
  using H4 = u4;
  using H = H4;
  using M = PHF_METADATA;
  using module = phf::cuhip::modules<E, H>;
  using phf_module = phf::cuhip::modules<E, H>;
  using Header = phf_header;

  phf_header header;

  Buf* buf;

  float _time_book{0.0}, _time_lossless{0.0};
  float time_book() const { return _time_book; }
  float time_codec() const { return _time_lossless; }
  float time_lossless() const { return _time_lossless; }
  size_t inlen() const { return len; };

  GPU_EVENT event_start, event_end;

  size_t pardeg, sublen;
  int numSMs;
  size_t len;
  static constexpr u2 max_bklen = 1024;
  u2 rt_bklen;

  GPU_unique_hptr<u4[]> h_hist;

  phf_dtype const in_dtype;

  // TODO Is specifying inlen when constructing proper?
  HuffmanCodec(size_t const inlen, int const pardeg, bool debug = false);
  ~HuffmanCodec();

  HuffmanCodec* buildbook(u4* d_hist_ext, u2 const rt_bklen, phf_stream_t);
  // alternatively, it can force check the input array
  HuffmanCodec* encode(E*, size_t const, PHF_BYTE**, size_t*, phf_stream_t);
  HuffmanCodec* decode(PHF_BYTE*, E*, phf_stream_t, bool = true);
  HuffmanCodec* clear_buffer();
  HuffmanCodec* dump_internal_data(std::string, std::string);

 private:
  void make_metadata();
};

struct HuffmanHelper {
  static const int BLOCK_DIM_ENCODE = 256;
  static const int BLOCK_DIM_DEFLATE = 256;

  static const int ENC_SEQUENTIALITY = 4;  // empirical
  static const int DEFLATE_CONSTANT = 4;   // deflate_chunk_constant
};

}  // namespace phf

#define HF_SPACE phf::Buf<E>
#define HF_STREAM void*

namespace phf {

template <typename E>
struct high_level {
  static int build_book(HF_SPACE* buf, u4* h_hist, u2 const runtime_bklen, HF_STREAM stream);

  static int encode(
      HF_SPACE* buf, E* in_data, size_t const data_len, uint8_t** out_encoded, size_t* encoded_len,
      phf_header& header, HF_STREAM stream);

  static int encode_ReVISIT_lite(
      HF_SPACE* buf, E* in_data, size_t const data_len, uint8_t** out_encoded, size_t* encoded_len,
      phf_header& header, HF_STREAM stream);

  static int decode(
      HF_SPACE* buf, phf_header& header, PHF_BYTE* in_encoded, E* out_decoded, HF_STREAM stream);
};

}  // namespace phf

#endif /* HF_HL_HH */
