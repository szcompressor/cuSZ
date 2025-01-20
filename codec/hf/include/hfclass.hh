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

#include <memory>

#include "hf.h"

namespace phf {

template <typename E>
class HuffmanCodec {
 private:
  using SYM = E;

  struct Buf;
  struct impl;
  std::unique_ptr<impl> pimpl;

 public:
  using H4 = u4;
  using H = H4;
  using M = PHF_METADATA;

  phf_dtype const in_dtype;

  // TODO Is specifying inlen when constructing proper?
  HuffmanCodec(size_t const inlen, int const bklen, int const pardeg, bool debug = false);
  ~HuffmanCodec();

  float time_book() const;
  float time_lossless() const;
  size_t inlen() const;

  // TODO check d_hist_ext boundary
  HuffmanCodec* buildbook(u4* d_hist_ext, phf_stream_t);
  // TODO inlen is unnecessary
  // alternatively, it can force check the input array
  HuffmanCodec* encode(E*, size_t const, PHF_BYTE**, size_t*, phf_stream_t);
  HuffmanCodec* decode(PHF_BYTE*, E*, phf_stream_t, bool = true);
  HuffmanCodec* clear_buffer();
  HuffmanCodec* dump_internal_data(std::string, std::string);
};

struct HuffmanHelper {
  static const int BLOCK_DIM_ENCODE = 256;
  static const int BLOCK_DIM_DEFLATE = 256;

  static const int ENC_SEQUENTIALITY = 4;  // empirical
  static const int DEFLATE_CONSTANT = 4;   // deflate_chunk_constant
};

}  // namespace phf

#endif /* DAB559E7_A5C1_4342_B17E_17C31DA96EEF */
