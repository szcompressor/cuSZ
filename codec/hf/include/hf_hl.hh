// High-level HF codec interface

#ifndef HF_HL_HH
#define HF_HL_HH

#include "cusz/type.h"
#include "hf.h"
#include "hfr-pbk_ver.hh"

namespace phf {

template <typename E>
struct Buf;  // full def in hf_buf.hh (private)

using H4 = u4;
using M = PHF_METADATA;
using HF_STREAM = void*;

#define PHF_BUF phf::Buf<E>
#define PHF_STREAM void*

// HF + HFR (detailed in namespace phf::dispatch)
template <typename E>
struct high_level {
  static int build_book(PHF_BUF* buf, u4* h_hist, u2 const runtime_bklen, PHF_STREAM stream);

  // clang-format off
  // HF{,_r1,_r2}
  static int HF_encode (PHF_BUF* buf, E* in_data, size_t const data_len, u1** out_encoded, size_t* encoded_len, phf_header& header, PHF_STREAM stream, psz_codec variant = HF, float* opt_ms_encoder = nullptr, float* opt_ms_lago = nullptr);
  static int HF_decode (PHF_BUF* buf, phf_header& header, PHF_BYTE* in_encoded, E* out_decoded, PHF_STREAM stream, psz_codec variant = HF);
  // HFR{,_PBK_Compat,_PBK_GO}
  static int HFR_encode(PHF_BUF* buf, E* in_data, size_t const data_len, u1** out_encoded, size_t* encoded_len, phf_header& header, PHF_STREAM stream, psz_codec variant, float* opt_ms_encoder = nullptr, float* opt_ms_lago = nullptr, HFR_Opts opts = {});
  static int HFR_decode(PHF_BUF* buf, phf_header& header, PHF_BYTE* in_encoded, E* out_decoded, PHF_STREAM stream, psz_codec variant);
  // clang-format on
};

}  // namespace phf

#undef PHF_BUF
#undef PHF_STREAM

#endif /* HF_HL_HH */
