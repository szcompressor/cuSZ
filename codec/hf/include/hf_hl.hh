// High-level Huffman encoding/decoding interface

#ifndef HF_HL_HH
#define HF_HL_HH

#include "c_type.h"
#include "hf.h"

namespace phf {

template <typename E>
struct Buf;  // forward declearation, full definition in hf_buf.hh (private)

using H4 = u4;
using M = PHF_METADATA;
using HF_STREAM = void*;

#define PHF_BUF phf::Buf<E>
#define PHF_STREAM void*

template <typename E>
struct high_level {
  static int build_book(PHF_BUF* buf, u4* h_hist, u2 const runtime_bklen, PHF_STREAM stream);

  static int encode(
      PHF_BUF* buf, E* in_data, size_t const data_len, u1** out_encoded, size_t* encoded_len,
      phf_header& header, PHF_STREAM stream);

  static int encode_HFR(
      PHF_BUF* buf, E* in_data, size_t const data_len, u1** out_encoded, size_t* encoded_len,
      phf_header& header, PHF_STREAM stream);

  static int decode(
      PHF_BUF* buf, phf_header& header, PHF_BYTE* in_encoded, E* out_decoded, PHF_STREAM stream);
};

}  // namespace phf

#undef PHF_BUF
#undef PHF_STREAM

#endif /* HF_HL_HH */
