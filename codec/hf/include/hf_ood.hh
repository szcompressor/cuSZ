#ifndef PHF_OOD_HH
#define PHF_OOD_HH

#include <memory>

#include "hf.h"
#include "hf_bk.h"
#include "hf_buf.hh"
#include "hf_kernels.hh"
#include "hf_ood.hh"
#include "hf_type.h"
#include "mem/cxx_sp_gpu.h"
#include "phf_array.hh"
#include "utils/io.hh"
#include "utils/timer.hh"

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

#endif /* PHF_OOD_HH */
