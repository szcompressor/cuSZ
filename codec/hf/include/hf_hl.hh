#include "c_type.h"
#include "hf_buf.hh"
#include "hf_type.h"

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