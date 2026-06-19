#include "hf_hl.hh"

#include <cstdio>
#include <type_traits>
#include <vector>

#include "hf_buf.hh"
#include "hf_impl.hh"
#include "hfr-pbk.hh"
#include "hfr-pbk_decoder.hh"
#include "hfr.hh"
#include "mem/cxx_backends.h"

#define PHF_ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_encoded + header.entry[PHFHEADER_##SYM])

// device-side prebuilt {book, reverse}: accessors (pbk25_r128_d.cu).
extern "C" void* pbk25_r128_book_d_ptr();
extern "C" void* pbk25_r128_rvbk_d_ptr();

using H4 = u4;
using M = PHF_METADATA;

namespace phf {

template <typename E>
using phf_module = cuhip::modules<E, H4>;

namespace dispatch {

template <typename E>
int encode_hf(
    Buf<E>* buf, E* in, size_t const len, uint8_t** out, size_t* outlen, phf_header& header,
    hf_stream_t stream)
{
  size_t _total_nbit = 0, _total_ncell = 0;  // wrapper writes size_t; header keeps u4 ncell
  phf_module<E>::GPU_coarse_encode(
      in, len, buf->book_d(), buf->rt_bklen(), buf->num_sms(), {buf->sublen(), buf->pardeg()},
      buf->scratch_d(), buf->par_nbit_d(), buf->par_nbit_h(), buf->par_ncell_d(),
      buf->par_ncell_h(), buf->par_entry_d(), buf->par_entry_h(), buf->bitstream_d(),
      buf->bitstream_max_len(), &_total_nbit, &_total_ncell, stream);

  sync_by_stream(stream);
  header.total_ncell = (u4)_total_ncell;

  {
    M nbyte[PHFHEADER_END];
    buf->update_header(header);
    buf->calc_offset(header, nbyte);
  }
  buf->memcpy_merge(header, stream);

  *out = buf->encoded_d();
  *outlen = phf_encoded_bytes(&header);
  return 0;
}

// HFr1: same ph1+ph2 as encode_hf, then LAGO-concat replaces ph3+ph4.
template <typename E>
int encode_hf_rev1(
    Buf<E>* buf, E* in, size_t const len, uint8_t** out, size_t* outlen, phf_header& header,
    hf_stream_t stream, float* opt_ms_encoder, float* opt_ms_lago)
{
  constexpr int ConcatBlockDim = 128;
  using Concat = phf::concat_via_scatter_ppc<ConcatBlockDim>;
  const phf::par_config hfpar{buf->sublen(), buf->pardeg()};

  const bool want_timing = opt_ms_encoder or opt_ms_lago;
  auto e0 = (cudaEvent_t)buf->timing_event(0);
  auto e1 = (cudaEvent_t)buf->timing_event(1);
  auto e2 = (cudaEvent_t)buf->timing_event(2);
  if (want_timing) cudaEventRecord(e0, (cudaStream_t)stream);

  phf_module<E>::GPU_coarse_enc_ph1(
      in, len, buf->book_d(), buf->rt_bklen(), buf->num_sms(), buf->scratch_d(), stream);
  phf_module<E>::GPU_coarse_enc_ph2(
      buf->scratch_d(), len, hfpar, buf->scratch_d(), buf->par_nbit_d(), buf->par_ncell_d(),
      stream);

  if (want_timing) cudaEventRecord(e1, (cudaStream_t)stream);

  Concat::GPU_kernel(
      buf->par_ncell_d(), buf->par_entry_d(), (u4 const*)buf->scratch_d(), (u4*)buf->bitstream_d(),
      (u4)hfpar.sublen, (int)hfpar.pardeg, buf->scan_partial_aggregate_d(),
      buf->scan_incl_prefix_d(), buf->scan_tile_status_d(), buf->total_ncell_d(), stream);

  if (want_timing) cudaEventRecord(e2, (cudaStream_t)stream);

  u4 h_total_ncell = 0;
  memcpy_allkinds_async<D2H>(&h_total_ncell, buf->total_ncell_d(), 1, stream);
  memcpy_allkinds_async<D2H>(buf->par_entry_h(), buf->par_entry_d(), hfpar.pardeg, stream);
  sync_by_stream(stream);

  header.total_ncell = h_total_ncell;

  {
    M nbyte[PHFHEADER_END];
    buf->update_header(header);
    buf->calc_offset(header, nbyte);
  }
  buf->memcpy_merge(header, stream);

  *out = buf->encoded_d();
  *outlen = phf_encoded_bytes(&header);

  if (want_timing) {
    if (opt_ms_encoder) cudaEventElapsedTime(opt_ms_encoder, e0, e1);
    if (opt_ms_lago) cudaEventElapsedTime(opt_ms_lago, e1, e2);
  }
  return 0;
}

// HFr2: as _r1, with per-block metadata as AoS bheader_backport[].
template <typename E>
int encode_hf_rev2(
    Buf<E>* buf, E* in, size_t const len, uint8_t** out, size_t* outlen, phf_header& header,
    hf_stream_t stream, float* opt_ms_encoder, float* opt_ms_lago)
{
  constexpr int ConcatBlockDim = 128;
  using Concat = phf::concat_via_scatter_ppc<ConcatBlockDim>;
  const phf::par_config hfpar{buf->sublen(), buf->pardeg()};

  const bool want_timing = opt_ms_encoder or opt_ms_lago;
  auto e0 = (cudaEvent_t)buf->timing_event(0);
  auto e1 = (cudaEvent_t)buf->timing_event(1);
  auto e2 = (cudaEvent_t)buf->timing_event(2);
  if (want_timing) cudaEventRecord(e0, (cudaStream_t)stream);

  phf_module<E>::GPU_coarse_enc_ph1(
      in, len, buf->book_d(), buf->rt_bklen(), buf->num_sms(), buf->scratch_d(), stream);
  phf_module<E>::GPU_coarse_enc_ph2(
      buf->scratch_d(), len, hfpar, buf->scratch_d(), buf->par_nbit_d(), buf->par_ncell_d(),
      stream);

  if (want_timing) cudaEventRecord(e1, (cudaStream_t)stream);

  Concat::GPU_kernel(
      buf->par_ncell_d(), buf->par_entry_d(), (u4 const*)buf->scratch_d(), (u4*)buf->bitstream_d(),
      (u4)hfpar.sublen, (int)hfpar.pardeg, buf->scan_partial_aggregate_d(),
      buf->scan_incl_prefix_d(), buf->scan_tile_status_d(), buf->total_ncell_d(), stream);

  if (want_timing) cudaEventRecord(e2, (cudaStream_t)stream);

  phf::module::pack_bheader_backport::GPU_kernel(
      buf->par_nbit_d(), buf->par_entry_d(), buf->hf_rev2_header_d(), (int)hfpar.pardeg,
      (int)sizeof(H4), stream);

  u4 h_total_ncell = 0;
  memcpy_allkinds_async<D2H>(&h_total_ncell, buf->total_ncell_d(), 1, stream);
  memcpy_allkinds_async<D2H>(buf->par_entry_h(), buf->par_entry_d(), hfpar.pardeg, stream);
  sync_by_stream(stream);

  header.total_ncell = h_total_ncell;

  buf->set_use_hf_rev2_header(true);

  {
    M nbyte[PHFHEADER_END];
    buf->update_header(header);
    buf->calc_offset(header, nbyte);
  }
  buf->memcpy_merge(header, stream);

  *out = buf->encoded_d();
  *outlen = phf_encoded_bytes(&header);

  if (want_timing) {
    if (opt_ms_encoder) cudaEventElapsedTime(opt_ms_encoder, e0, e1);
    if (opt_ms_lago) cudaEventElapsedTime(opt_ms_lago, e1, e2);
  }
  return 0;
}

template <typename Ein, typename Eout = Ein>
int decode_hf(
    Buf<Ein>* buf, phf_header& header, uint8_t* in_encoded, Eout* out_decoded, hf_stream_t stream)
{
  phf_module<Ein>::template GPU_coarse_decode<Eout>(
      PHF_ACCESSOR(BITSTREAM, H4), PHF_ACCESSOR(RVBK, PHF_BYTE), buf->rvbk_bytes(),
      PHF_ACCESSOR(PAR_NBIT, M), PHF_ACCESSOR(PAR_ENTRY, M), header.sublen, header.pardeg,
      out_decoded, /*par_encid=*/nullptr, stream);
  return 0;
}

template <typename Ein, typename Eout = Ein>
int decode_hf_rev2(
    Buf<Ein>* buf, phf_header& header, uint8_t* in_encoded, Eout* out_decoded, hf_stream_t stream)
{
  // Unpack AoS bheader_backport[] -> par_nbit / par_entry, then GPU_coarse_decode.
  auto packed = reinterpret_cast<u4 const*>(in_encoded + header.entry[PHFHEADER_HF_REV2_HEADER]);
  phf::module::unpack_bheader_backport::GPU_kernel(
      packed, buf->par_nbit_d(), buf->par_entry_d(), (int)header.pardeg, (int)sizeof(H4), stream);

  phf_module<Ein>::template GPU_coarse_decode<Eout>(
      reinterpret_cast<H4*>(in_encoded + header.entry[PHFHEADER_BITSTREAM]),
      reinterpret_cast<PHF_BYTE*>(in_encoded + header.entry[PHFHEADER_RVBK]), buf->rvbk_bytes(),
      buf->par_nbit_d(), buf->par_entry_d(), header.sublen, header.pardeg, out_decoded,
      /*par_encid=*/nullptr, stream);
  return 0;
}

}  // namespace dispatch

// Shared encode pipeline for HFR-v2 / PBKC / PBKGO.
template <typename E, typename LaunchEnc, typename LaunchAggregate>
static int _HFR_common_enc(
    Buf<E>* buf, size_t const len, uint8_t** out, size_t* outlen, phf_header& header,
    hf_stream_t stream, LaunchEnc&& launch_encode, LaunchAggregate&& launch_aggregate,
    float* opt_ms_encoder = nullptr, float* opt_ms_lago = nullptr)
{
  using K = psz::HFR_PBK_Constants;
  buf->set_rt_bklen(K::MaxDictsize);

  const bool want_timing = opt_ms_encoder or opt_ms_lago;
  auto e0 = (cudaEvent_t)buf->timing_event(0);
  auto e1 = (cudaEvent_t)buf->timing_event(1);
  auto e2 = (cudaEvent_t)buf->timing_event(2);
  if (want_timing) cudaEventRecord(e0, (cudaStream_t)stream);

  launch_encode(buf->pbk_headers_d());

  if (want_timing) cudaEventRecord(e1, (cudaStream_t)stream);

  launch_aggregate();

  if (want_timing) cudaEventRecord(e2, (cudaStream_t)stream);

  u4 h_total_ncell = 0;
  memcpy_allkinds_async<D2H>(&h_total_ncell, buf->total_ncell_d(), 1, stream);
  sync_by_stream(stream);

  header.total_ncell = h_total_ncell;

  {
    M nbyte[PHFHEADER_END];
    buf->update_header(header);
    buf->calc_offset(header, nbyte);
  }
  buf->memcpy_merge(header, stream);

  *out = buf->encoded_d();
  *outlen = phf_encoded_bytes(&header);

  if (want_timing) {
    if (opt_ms_encoder) cudaEventElapsedTime(opt_ms_encoder, e0, e1);
    if (opt_ms_lago) cudaEventElapsedTime(opt_ms_lago, e1, e2);
  }
  return 0;
}

namespace dispatch {

// HFR v2: PBKC-backport, but with single runtime-built book instread of pbk
template <typename E>
int encode_hfr_v2(
    Buf<E>* buf, E* in, size_t const len, uint8_t** out, size_t* outlen, phf_header& header,
    hf_stream_t stream, float* opt_ms_encoder, float* opt_ms_lago, HFR_Opts opts)
{
  if constexpr (sizeof(E) > 2) {
    (void)buf, (void)in, (void)len, (void)out, (void)outlen, (void)header, (void)stream,
        (void)opt_ms_encoder, (void)opt_ms_lago, (void)opts;
    return PHF_NOT_IMPLEMENTED;
  }
  else {
    const int reduce_times = opts.reduce_times;
    const RMerge rm = opts.rm;
    const SMerge sm = opts.sm;
    constexpr int ConcatBlockDim = 128;
    using K = psz::HFR_PBK_Constants;
    using ConcatFuture = phf::_future_concat_via_scatter<E, ConcatBlockDim>;
    buf->set_use_prebuilt_rvbk(false);  // HFR ships runtime rvbk in archive.
    const size_t pardeg = (len - 1) / K::BlockSize + 1;
    auto launch_enc = [&]<int RT>(std::integral_constant<int, RT>, auto* hdrs) {
      using Enc = phf::module::HFR_encoder<E, K::Magnitude, RT>;
      Enc::template GPU_kernel_v2<K::Radius>(
          in, len, buf->book_d(), buf->bitstream_d(), hdrs, stream, rm, sm);
    };
    auto launch_aggregate = [&]() {
      ConcatFuture::GPU_kernel(
          buf->pbk_headers_d(), buf->par_entry_d(), buf->bitstream_d(), buf->packed_d(),
          buf->pbk_packed_headers_d(), (u4)sizeof(H4), (u4)K::StridePerBlockWords, (int)pardeg,
          buf->scan_partial_aggregate_d(), buf->scan_incl_prefix_d(), buf->scan_tile_status_d(),
          buf->total_ncell_d(), stream);
    };
    switch (reduce_times) {
      case 0:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 0>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago);
      case 1:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 1>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago);
      case 2:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 2>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago);
      case 3:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 3>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago);
      case 4:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 4>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago);
      default: return PHF_NOT_IMPLEMENTED;
    }
  }
}

// HFR v3: like v2 but pick a PBK, with preferred low #rmerge
template <typename E>
int encode_hfr_v3(
    Buf<E>* buf, E* in, size_t const len, uint8_t** out, size_t* outlen, phf_header& header,
    hf_stream_t stream, float* opt_ms_encoder, float* opt_ms_lago, HFR_Opts opts)
{
  if constexpr (sizeof(E) > 2) {
    (void)buf, (void)in, (void)len, (void)out, (void)outlen, (void)header, (void)stream,
        (void)opt_ms_encoder, (void)opt_ms_lago, (void)opts;
    return PHF_NOT_IMPLEMENTED;
  }
  else {
    const int reduce_times = opts.reduce_times;
    const RMerge rm = opts.rm;
    const SMerge sm = opts.sm;
    constexpr int ConcatBlockDim = 128;
    using K = psz::HFR_PBK_Constants;
    using ConcatFuture = phf::_future_concat_via_scatter<E, ConcatBlockDim>;
    buf->set_use_prebuilt_rvbk(true);  // rvbk stays baked-in; decode indexes it by global id.
    buf->set_use_global_encid(true);   // patch the picked book id into the archive header.
    const size_t pardeg = (len - 1) / K::BlockSize + 1;
    auto launch_enc = [&]<int RT>(std::integral_constant<int, RT>, auto* hdrs) {
      using Enc = phf::module::HFR_encoder<E, K::Magnitude, RT>;
      Enc::template GPU_kernel_v2<K::Radius>(
          in, len, buf->book_d(), buf->bitstream_d(), hdrs, stream, rm, sm);
    };
    auto launch_aggregate = [&]() {
      ConcatFuture::GPU_kernel(
          buf->pbk_headers_d(), buf->par_entry_d(), buf->bitstream_d(), buf->packed_d(),
          buf->pbk_packed_headers_d(), (u4)sizeof(H4), (u4)K::StridePerBlockWords, (int)pardeg,
          buf->scan_partial_aggregate_d(), buf->scan_incl_prefix_d(), buf->scan_tile_status_d(),
          buf->total_ncell_d(), stream);
    };
    switch (reduce_times) {
      case 0:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 0>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago);
      case 1:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 1>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago);
      case 2:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 2>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago);
      case 3:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 3>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago);
      case 4:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 4>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago);
      default: return PHF_NOT_IMPLEMENTED;
    }
  }
}

template <typename E>
int encode_hfr_pbkc(
    Buf<E>* buf, E* in, size_t const len, uint8_t** out, size_t* outlen, phf_header& header,
    hf_stream_t stream, float* opt_ms_encoder, float* opt_ms_lago, HFR_Opts opts)
{
  if constexpr (sizeof(E) > 2) {
    (void)buf, (void)in, (void)len, (void)out, (void)outlen, (void)header, (void)stream,
        (void)opt_ms_encoder, (void)opt_ms_lago, (void)opts;
    return PHF_NOT_IMPLEMENTED;
  }
  else {
    const int reduce_times = opts.reduce_times;
    const RMerge rm = opts.rm;
    const SMerge sm = opts.sm;
    constexpr int ConcatBlockDim = 128;
    using K = psz::HFR_PBK_Constants;
    using ConcatFuture = phf::_future_concat_via_scatter<E, ConcatBlockDim>;
    buf->set_use_prebuilt_rvbk(true);
    const size_t pardeg = (len - 1) / K::BlockSize + 1;
    auto launch_enc = [&]<int RT>(std::integral_constant<int, RT>, auto* hdrs) {
      using Enc = phf::module::HFR_PBKC_encode<E, K::Magnitude, RT, H4, K::Radius>;
      Enc::GPU_kernel(
          in, len, (H4*)pbk25_r128_book_d_ptr(), buf->bitstream_d(), hdrs, stream, rm, sm);
    };
    auto launch_aggregate = [&]() {
      ConcatFuture::GPU_kernel(
          buf->pbk_headers_d(), buf->par_entry_d(), buf->bitstream_d(), buf->packed_d(),
          buf->pbk_packed_headers_d(), (u4)sizeof(H4), (u4)K::StridePerBlockWords, (int)pardeg,
          buf->scan_partial_aggregate_d(), buf->scan_incl_prefix_d(), buf->scan_tile_status_d(),
          buf->total_ncell_d(), stream);
    };
    switch (reduce_times) {
      case 0:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 0>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago);
      case 1:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 1>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago);
      case 2:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 2>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago);
      case 3:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 3>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago);
      case 4:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 4>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago);
      default: return PHF_NOT_IMPLEMENTED;
    }
  }
}

template <typename E>
int encode_hfr_pbkgo(
    Buf<E>* buf, E* in, size_t const len, uint8_t** out, size_t* outlen, phf_header& header,
    hf_stream_t stream, float* opt_ms_encoder, float* opt_ms_lago, HFR_Opts opts)
{
  if constexpr (sizeof(E) > 2) {
    (void)buf, (void)in, (void)len, (void)out, (void)outlen, (void)header, (void)stream,
        (void)opt_ms_encoder, (void)opt_ms_lago, (void)opts;
    return PHF_NOT_IMPLEMENTED;
  }
  else {
    const int reduce_times = opts.reduce_times;
    const RMerge rm = opts.rm;
    const SMerge sm = opts.sm;
    using K = psz::HFR_PBK_Constants;
    int rt = reduce_times;
    if (rt < 2) {
      fprintf(stderr, "[phf::warn] HFR-PBKGO falls back to rmerge-count >=2.\n", rt);
      rt = 2;
    }
    buf->set_use_prebuilt_rvbk(true);
    buf->set_use_pbkgo(true);
    auto launch_enc = [&]<int RT>(std::integral_constant<int, RT>, auto* /*unused*/) {
      using Enc = phf::module::HFR_PBKGO_encode<E, K::Magnitude, RT, H4, K::Radius>;
      Enc::GPU_kernel(
          in, len, (H4*)pbk25_r128_book_d_ptr(), buf->bitstream_d(), buf->pbk_packed_headers_d(),
          buf->total_ncell_d(), buf->pbkgo_state_d(), buf->pbkgo_max_resident_blocks(), stream, rm,
          sm);
    };
    auto launch_aggregate = []() { /* no-op: encoder emitted everything */ };
    switch (rt) {
      case 2:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 2>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago);
      case 3:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 3>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago);
      case 4:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 4>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago);
      default: return PHF_NOT_IMPLEMENTED;
    }
  }
}

// Unified HFR-family decode: one GPU_kernel; variant picks (Storage, reverse-book source).
template <typename Ein, typename Eout = Ein>
int decode_hfr(
    Buf<Ein>* buf, phf_header& header, uint8_t* in_encoded, Eout* out_decoded, hf_stream_t stream,
    psz_codec variant)
{
  if constexpr (sizeof(Ein) > 2) {
    (void)buf, (void)header, (void)in_encoded, (void)out_decoded, (void)stream, (void)variant;
    return PHF_NOT_IMPLEMENTED;
  }
  else {
    using K = psz::HFR_PBK_Constants;
    constexpr int RvbkBytesPerBook = (int)K::RvbkBytesPerBook;  // 512

    auto bs_ptr = (H4*)(in_encoded + header.entry[PHFHEADER_BITSTREAM]);
    auto packed_headers = (uint32_t const*)(in_encoded + header.entry[PHFHEADER_PBK_HEADERS]);
    const auto bs_bytes = header.total_ncell * sizeof(H4);
    const int pardeg = (int)header.pardeg;

    if (variant == HFR) {
      // runtime-built reverse book (Storage=Ein), one book for all blocks.
      auto rvbk = (uint8_t*)(in_encoded + header.entry[PHFHEADER_RVBK]);
      const int rvbk_bytes = (int)(header.entry[PHFHEADER_RVBK + 1] - header.entry[PHFHEADER_RVBK]);
      phf::module::HFR_PBK_decoder<Ein, H4, Ein>::template GPU_kernel<Eout>(
          bs_ptr, bs_bytes, rvbk, rvbk_bytes, packed_headers, pardeg, header.ori_len, out_decoded,
          stream);
    }
    else {
      // prebuilt pbk25_r128 pool (Storage=u1); V3 offsets to its single g_encid book.
      auto rvbk = (uint8_t*)pbk25_r128_rvbk_d_ptr() +
                  (variant == HFR_V3 ? (size_t)header.g_encid * RvbkBytesPerBook : 0);
      phf::module::HFR_PBK_decoder<Ein, H4, u1>::template GPU_kernel<Eout>(
          bs_ptr, bs_bytes, rvbk, RvbkBytesPerBook, packed_headers, pardeg, header.ori_len,
          out_decoded, stream);
    }

    sync_by_stream(stream);
    return 0;
  }
}

#undef PHF_ACCESSOR

}  // namespace dispatch

template <typename E>
int high_level<E>::HF_build_book(
    phf::Buf<E>* buf, u4* h_hist, u2 const rt_bklen, hf_stream_t stream)
{
  buf->set_rt_bklen(rt_bklen);

  phf_CPU_build_canonized_codebook_v2<E, H4>(
      h_hist, rt_bklen, buf->book_h(), buf->rvbk_h(), buf->rvbk_bytes());
  // clang-format off
  memcpy_allkinds_async<H2D>(buf->book_d(), buf->book_h(), rt_bklen, (cudaStream_t)stream);
  memcpy_allkinds_async<H2D>(buf->rvbk_d(), buf->rvbk_h(), buf->rvbk_bytes(), (cudaStream_t)stream);
  // clang-format on
  return 0;
}

// HFR-v3 book source: pick one global PBK book from the histogram (GPU-only),
// replacing the CPU canonical-book build of HF_build_book().
template <typename E>
int high_level<E>::HFR_pick_pbk(
    phf::Buf<E>* buf, u4* hist_d, u2 const bklen, size_t const len, hf_stream_t stream)
{
  buf->set_rt_bklen(psz::HFR_PBK_Constants::MaxDictsize);
  if constexpr (sizeof(E) <= 2)
    phf::module::HFR_pick_pbk(
        hist_d, (u4)bklen, len, (u4*)pbk25_r128_book_d_ptr(), buf->book_d(), buf->pick_encid_d(),
        stream);
  return 0;
}

template <typename E>
int high_level<E>::HF_encode(
    Buf<E>* buf, E* in, size_t const len, uint8_t** out, size_t* outlen, phf_header& header,
    hf_stream_t stream, psz_codec variant, float* opt_ms_encoder, float* opt_ms_lago)
{
  switch (variant) {
    case HF:
      (void)opt_ms_encoder;
      (void)opt_ms_lago;
      return dispatch::encode_hf<E>(buf, in, len, out, outlen, header, stream);
    case HFr1:
      return dispatch::encode_hf_rev1<E>(
          buf, in, len, out, outlen, header, stream, opt_ms_encoder, opt_ms_lago);
    case HFr2:
      return dispatch::encode_hf_rev2<E>(
          buf, in, len, out, outlen, header, stream, opt_ms_encoder, opt_ms_lago);
    default: return PHF_NOT_IMPLEMENTED;
  }
}

template <typename E>
template <typename Eout>
int high_level<E>::HF_decode(
    Buf<E>* buf, phf_header& header, uint8_t* in_encoded, Eout* out_decoded, hf_stream_t stream,
    psz_codec variant)
{
  // HF{,_r1}: same layout, so same decoder
  switch (variant) {
    case HF:
    case HFr1: return dispatch::decode_hf<E, Eout>(buf, header, in_encoded, out_decoded, stream);
    case HFr2:
      return dispatch::decode_hf_rev2<E, Eout>(buf, header, in_encoded, out_decoded, stream);
    default: return PHF_NOT_IMPLEMENTED;
  }
}

template <typename E>
int high_level<E>::HFR_encode(
    Buf<E>* buf, E* in, size_t const len, uint8_t** out, size_t* outlen, phf_header& header,
    hf_stream_t stream, psz_codec variant, float* opt_ms_encoder, float* opt_ms_lago,
    HFR_Opts opts)
{
  switch (variant) {
    case HFR:
      return dispatch::encode_hfr_v2<E>(
          buf, in, len, out, outlen, header, stream, opt_ms_encoder, opt_ms_lago, opts);
    case HFR_PBKC:
      return dispatch::encode_hfr_pbkc<E>(
          buf, in, len, out, outlen, header, stream, opt_ms_encoder, opt_ms_lago, opts);
    case HFR_PBKGO:
      return dispatch::encode_hfr_pbkgo<E>(
          buf, in, len, out, outlen, header, stream, opt_ms_encoder, opt_ms_lago, opts);
    case HFR_V3:
      return dispatch::encode_hfr_v3<E>(
          buf, in, len, out, outlen, header, stream, opt_ms_encoder, opt_ms_lago, opts);
    case HFR_PBKF: return PHF_NOT_IMPLEMENTED;
    default: return PHF_NOT_IMPLEMENTED;
  }
}

template <typename E>
template <typename Eout>
int high_level<E>::HFR_decode(
    Buf<E>* buf, phf_header& header, uint8_t* in_encoded, Eout* out_decoded, hf_stream_t stream,
    psz_codec variant)
{
  switch (variant) {
    case HFR:
    case HFR_PBKC:
    case HFR_PBKGO:
    case HFR_V3:
      return dispatch::decode_hfr<E, Eout>(buf, header, in_encoded, out_decoded, stream, variant);
    case HFR_PBKF: return PHF_NOT_IMPLEMENTED;
    default: return PHF_NOT_IMPLEMENTED;
  }
}

}  // namespace phf

template struct phf::high_level<u1>;
template struct phf::high_level<u2>;
template struct phf::high_level<u4>;

template int phf::high_level<u2>::HF_decode<u2>(
    phf::Buf<u2>*, phf_header&, uint8_t*, u2*, hf_stream_t, psz_codec);
template int phf::high_level<u2>::HFR_decode<u2>(
    phf::Buf<u2>*, phf_header&, uint8_t*, u2*, hf_stream_t, psz_codec);

template int phf::high_level<u2>::HF_decode<f4>(
    phf::Buf<u2>*, phf_header&, uint8_t*, f4*, hf_stream_t, psz_codec);
template int phf::high_level<u2>::HF_decode<f8>(
    phf::Buf<u2>*, phf_header&, uint8_t*, f8*, hf_stream_t, psz_codec);
template int phf::high_level<u2>::HFR_decode<f4>(
    phf::Buf<u2>*, phf_header&, uint8_t*, f4*, hf_stream_t, psz_codec);
template int phf::high_level<u2>::HFR_decode<f8>(
    phf::Buf<u2>*, phf_header&, uint8_t*, f8*, hf_stream_t, psz_codec);
