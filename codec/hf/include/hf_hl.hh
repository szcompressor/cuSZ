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
using hf_stream_t = void*;

constexpr bool needs_book(psz_codec c)
{
  switch (c) {
    case HF:
    case HF_r2:
    case HFR:
    case HFR_V3:
    case HFR_V4: return true;
    default: return false;
  }
}

#define BUF phf::Buf<E>

// HF + HFR (detailed in namespace phf::dispatch)
template <typename E>
struct high_level {
  // clang-format off
  static int HF_build_book(BUF* buf, u2 const runtime_bklen, hf_stream_t s, u4* h_hist = nullptr);
  static int histogram(BUF* buf, E* in_data, size_t const data_len, u2 const bklen, hf_stream_t s, psz_hist hist = HistGeneric);
  static int make_book(BUF* buf, E* in_data, size_t const data_len, u2 const bklen, hf_stream_t s, psz_hist hist = HistGeneric);
  static int HFR_pick_pbk(BUF* buf, E* in_data, size_t const data_len, u2 const bklen, hf_stream_t s, psz_hist hist = HistGeneric);

  // HF{,_r1,_r2}
  static int HF_encode (BUF* buf, E* in_data, size_t const data_len, u1** out_encoded, size_t* encoded_len, phf_header& header, hf_stream_t s, psz_codec variant = HF, float* opt_ms_encoder = nullptr, float* opt_ms_lago = nullptr);
  template <typename Eout = E> static int HF_decode (BUF* buf, phf_header& header, PHF_BYTE* in_encoded, Eout* out_decoded, hf_stream_t s, psz_codec variant = HF);
  // HFR{,_PBK_Compat,_PBK_GO}
  static int HFR_RTBK_encode(BUF* buf, E* in_data, size_t const data_len, u1** out_encoded, size_t* encoded_len, phf_header& header, hf_stream_t s, float* opt_ms_encoder = nullptr, float* opt_ms_lago = nullptr, HFR_Opts opts = {});
  static int HFR_PBK_encode(BUF* buf, E* in_data, size_t const data_len, u1** out_encoded, size_t* encoded_len, phf_header& header, hf_stream_t s, psz_codec variant, float* opt_ms_encoder = nullptr, float* opt_ms_lago = nullptr, HFR_Opts opts = {});
  template <typename Eout = E> static int HFR_decode(BUF* buf, phf_header& header, PHF_BYTE* in_encoded, Eout* out_decoded, hf_stream_t s, psz_codec variant, int magnitude = 10);
  template <typename Eout = E> static int HFD26_decode(BUF* buf, phf_header& header, PHF_BYTE* in_encoded, Eout* out_decoded, hf_stream_t s, psz_codec variant, int magnitude = 10);
  // clang-format on
};

}  // namespace phf

#undef BUF

#endif /* HF_HL_HH */
