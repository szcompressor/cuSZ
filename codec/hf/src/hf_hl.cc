#include "hf_hl.hh"

#include <cstdio>
#include <type_traits>
#include <vector>

#include "hf_buf.hh"
#include "hf_impl.hh"
#include "hfd26.hh"
#include "hfr-pbk.hh"
#include "hfr-pbk_decoder.hh"
#include "hfr.hh"
#include "mem/cxx_backends.h"

#define PHF_ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_encoded + header.entry[PHFHEADER_##SYM])

extern "C" void* pbk25_r128_book_d_ptr();
extern "C" void* pbk25_r128_rvbk_d_ptr();

using H4 = u4;
using M = PHF_METADATA;

namespace phf {

template <typename E>
using phf_module = cuhip::modules<E, H4>;

template <typename E, typename KC>
static constexpr u4 _hfr_stride_words()
{
  using BreakCell = psz::HFR_PBK_Breaks<psz::HFR_PBK_Constants::Radius>;
  return (u4)((KC::BlockSize * sizeof(E) + KC::MaxNumBreaks * sizeof(BreakCell) +
               KC::MaxUnpredBytes + sizeof(u4) - 1) /
              sizeof(u4));
}

template <typename E>
static u4 hfr_stride_words(int magnitude)
{
  if (magnitude >= 12) return _hfr_stride_words<E, psz::HFR_PBK_C12>();
  if (magnitude >= 11) return _hfr_stride_words<E, psz::HFR_PBK_C11>();
  return _hfr_stride_words<E, psz::HFR_PBK_Constants>();
}

namespace dispatch {

// HFr2: as _r1, with per-block metadata as AoS bheader_backport[].
template <typename E>
int encode_hf_rev2(
    Buf<E>* buf, E* in, size_t const len, u1** out, size_t* outlen, phf_header& header,
    hf_stream_t stream, float* opt_ms_encoder, float* opt_ms_lago)
{
  constexpr auto ConcatBlockDim = 128;
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
      buf->par_nbit_d(), buf->par_entry_d(), (u4*)buf->pbk_headers_d(), (int)hfpar.pardeg,
      (int)sizeof(H4), stream);

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

template <typename Ein, typename Eout = Ein>
int decode_hf_rev2(
    Buf<Ein>* buf, phf_header& header, u1* in_encoded, Eout* out_decoded, hf_stream_t stream)
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
    Buf<E>* buf, size_t const len, u1** out, size_t* outlen, phf_header& header,
    hf_stream_t stream, LaunchEnc&& launch_encode, LaunchAggregate&& launch_aggregate,
    float* opt_ms_encoder = nullptr, float* opt_ms_lago = nullptr, int pardeg_override = 0)
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
    // reconcile cases other than 1Ki
    if (pardeg_override > 0) header.pardeg = pardeg_override;
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
    Buf<E>* buf, E* in, size_t const len, u1** out, size_t* outlen, phf_header& header,
    hf_stream_t stream, float* opt_ms_encoder, float* opt_ms_lago, HFR_Opts opts)
{
  {
    const int magnitude = opts.magnitude;
    // failsafe: block has no greater than 1024 threads.
    const int reduce_times = magnitude >= 12   ? (opts.reduce_times < 2 ? 2 : opts.reduce_times)
                             : magnitude >= 11 ? (opts.reduce_times < 1 ? 1 : opts.reduce_times)
                                               : opts.reduce_times;
    constexpr auto ConcatBlockDim = 128;
    using K = psz::HFR_PBK_Constants;
    buf->set_use_prebuilt_rvbk(false);  // HFR ships runtime rvbk in archive.
    const size_t pardeg = (len - 1) / ((size_t)1u << magnitude) + 1;
    const u4 stride_words = hfr_stride_words<E>(magnitude);
    auto launch_enc = [&]<int RT>(std::integral_constant<int, RT>, auto* hdrs) {
      if (magnitude >= 12) {
        if constexpr (RT >= 2) {
          using Enc = phf::module::HFR_encoder<E, 12, RT>;
          Enc::GPU_kernel_v2(
              in, len, buf->book_d(), buf->bitstream_d(), (typename Enc::bheader_t*)hdrs,
              opts.block_outliers, stream);
        }
        return;
      }
      if (magnitude >= 11) {
        if constexpr (RT >= 1) {
          using Enc = phf::module::HFR_encoder<E, 11, RT>;
          Enc::GPU_kernel_v2(
              in, len, buf->book_d(), buf->bitstream_d(), (typename Enc::bheader_t*)hdrs,
              opts.block_outliers, stream);
        }
        return;
      }
      using Enc = phf::module::HFR_encoder<E, K::Magnitude, RT>;
      Enc::GPU_kernel_v2(
          in, len, buf->book_d(), buf->bitstream_d(), hdrs, opts.block_outliers, stream);
    };
    auto launch_aggregate = [&]() {
      auto concat = [&]<int M>(std::integral_constant<int, M>) {
        using Concat = phf::_future_concat_via_scatter<E, ConcatBlockDim, M>;
        Concat::GPU_kernel(
            (typename Concat::bheader_t*)buf->pbk_headers_d(), buf->par_entry_d(),
            buf->bitstream_d(), buf->packed_d(), buf->pbk_packed_headers_d(), (u4)sizeof(H4),
            stride_words, (int)pardeg, buf->scan_partial_aggregate_d(), buf->scan_incl_prefix_d(),
            buf->scan_tile_status_d(), buf->total_ncell_d(), stream);
      };
      if (magnitude >= 12)
        concat(std::integral_constant<int, 12>{});
      else if (magnitude >= 11)
        concat(std::integral_constant<int, 11>{});
      else
        concat(std::integral_constant<int, 10>{});
    };
    switch (reduce_times) {
      case 0:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 0>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      case 1:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 1>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      case 2:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 2>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      case 3:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 3>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      case 4:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 4>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      default: return PHF_NOT_IMPLEMENTED;
    }
  }
}

// HFR v3: like v2 but pick a PBK, with preferred low #rmerge
template <typename E>
int encode_hfr_v3(
    Buf<E>* buf, E* in, size_t const len, u1** out, size_t* outlen, phf_header& header,
    hf_stream_t stream, float* opt_ms_encoder, float* opt_ms_lago, HFR_Opts opts)
{
  {
    const int magnitude = opts.magnitude;
    // BlockDim = 2^(magnitude - RT): clamp RT so the launch stays <= 1024 threads.
    const int reduce_times = magnitude >= 12   ? (opts.reduce_times < 2 ? 2 : opts.reduce_times)
                             : magnitude >= 11 ? (opts.reduce_times < 1 ? 1 : opts.reduce_times)
                                               : opts.reduce_times;
    constexpr auto ConcatBlockDim = 128;
    using K = psz::HFR_PBK_Constants;
    buf->set_use_prebuilt_rvbk(true);
    buf->set_use_global_encid(true);
    const size_t pardeg = (len - 1) / ((size_t)1u << magnitude) + 1;
    const u4 stride_words = hfr_stride_words<E>(magnitude);
    auto launch_enc = [&]<int RT>(std::integral_constant<int, RT>, auto* hdrs) {
      if (magnitude >= 12) {
        if constexpr (RT >= 2) {
          using Enc = phf::module::HFR_encoder<E, 12, RT>;
          Enc::GPU_kernel_v2(
              in, len, buf->book_d(), buf->bitstream_d(), (typename Enc::bheader_t*)hdrs,
              opts.block_outliers, stream);
        }
        return;
      }
      if (magnitude >= 11) {
        if constexpr (RT >= 1) {
          using Enc = phf::module::HFR_encoder<E, 11, RT>;
          Enc::GPU_kernel_v2(
              in, len, buf->book_d(), buf->bitstream_d(), (typename Enc::bheader_t*)hdrs,
              opts.block_outliers, stream);
        }
        return;
      }
      using Enc = phf::module::HFR_encoder<E, K::Magnitude, RT>;
      Enc::GPU_kernel_v2(
          in, len, buf->book_d(), buf->bitstream_d(), hdrs, opts.block_outliers, stream);
    };
    auto launch_aggregate = [&]() {
      auto concat = [&]<int M>(std::integral_constant<int, M>) {
        using Concat = phf::_future_concat_via_scatter<E, ConcatBlockDim, M>;
        Concat::GPU_kernel(
            (typename Concat::bheader_t*)buf->pbk_headers_d(), buf->par_entry_d(),
            buf->bitstream_d(), buf->packed_d(), buf->pbk_packed_headers_d(), (u4)sizeof(H4),
            stride_words, (int)pardeg, buf->scan_partial_aggregate_d(), buf->scan_incl_prefix_d(),
            buf->scan_tile_status_d(), buf->total_ncell_d(), stream);
      };
      if (magnitude >= 12)
        concat(std::integral_constant<int, 12>{});
      else if (magnitude >= 11)
        concat(std::integral_constant<int, 11>{});
      else
        concat(std::integral_constant<int, 10>{});
    };
    switch (reduce_times) {
      case 0:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 0>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      case 1:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 1>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      case 2:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 2>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      case 3:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 3>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      case 4:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 4>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      default: return PHF_NOT_IMPLEMENTED;
    }
  }
}

// HFR v4: v3 book pick + prebuilt rvbk, but encoded on the PBKC kernel (single-book mode).
template <typename E>
int encode_hfr_v4(
    Buf<E>* buf, E* in, size_t const len, u1** out, size_t* outlen, phf_header& header,
    hf_stream_t stream, float* opt_ms_encoder, float* opt_ms_lago, HFR_Opts opts)
{
  {
    const int magnitude = opts.magnitude;
    // BlockDim = 2^(magnitude - RT): clamp RT so the launch stays <= 1024 threads.
    const int reduce_times = magnitude >= 12   ? (opts.reduce_times < 2 ? 2 : opts.reduce_times)
                             : magnitude >= 11 ? (opts.reduce_times < 1 ? 1 : opts.reduce_times)
                                               : opts.reduce_times;
    constexpr auto ConcatBlockDim = 128;
    using K = psz::HFR_PBK_Constants;
    buf->set_use_prebuilt_rvbk(true);
    buf->set_use_global_encid(true);
    const size_t pardeg = (len - 1) / ((size_t)1u << magnitude) + 1;
    const u4 stride_words = hfr_stride_words<E>(magnitude);
    auto launch_enc = [&]<int RT>(std::integral_constant<int, RT>, auto* hdrs) {
      if (magnitude >= 12) {
        if constexpr (RT >= 2) {
          using Enc = phf::module::HFR_V4_encode<E, 12, RT, H4, K::Radius>;
          Enc::GPU_kernel(
              in, len, buf->book_d(), buf->bitstream_d(), (typename Enc::header_t*)hdrs,
              opts.block_outliers, stream);
        }
        return;
      }
      if (magnitude >= 11) {
        if constexpr (RT >= 1) {
          using Enc = phf::module::HFR_V4_encode<E, 11, RT, H4, K::Radius>;
          Enc::GPU_kernel(
              in, len, buf->book_d(), buf->bitstream_d(), (typename Enc::header_t*)hdrs,
              opts.block_outliers, stream);
        }
        return;
      }
      using Enc = phf::module::HFR_V4_encode<E, K::Magnitude, RT, H4, K::Radius>;
      Enc::GPU_kernel(
          in, len, buf->book_d(), buf->bitstream_d(), hdrs, opts.block_outliers, stream);
    };
    auto launch_aggregate = [&]() {
      auto concat = [&]<int M>(std::integral_constant<int, M>) {
        using Concat = phf::_future_concat_via_scatter<E, ConcatBlockDim, M>;
        Concat::GPU_kernel(
            (typename Concat::bheader_t*)buf->pbk_headers_d(), buf->par_entry_d(),
            buf->bitstream_d(), buf->packed_d(), buf->pbk_packed_headers_d(), (u4)sizeof(H4),
            stride_words, (int)pardeg, buf->scan_partial_aggregate_d(), buf->scan_incl_prefix_d(),
            buf->scan_tile_status_d(), buf->total_ncell_d(), stream);
      };
      if (magnitude >= 12)
        concat(std::integral_constant<int, 12>{});
      else if (magnitude >= 11)
        concat(std::integral_constant<int, 11>{});
      else
        concat(std::integral_constant<int, 10>{});
    };
    switch (reduce_times) {
      case 0:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 0>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      case 1:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 1>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      case 2:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 2>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      case 3:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 3>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      case 4:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 4>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      default: return PHF_NOT_IMPLEMENTED;
    }
  }
}

template <typename E>
int encode_hfr_pbkc(
    Buf<E>* buf, E* in, size_t const len, u1** out, size_t* outlen, phf_header& header,
    hf_stream_t stream, float* opt_ms_encoder, float* opt_ms_lago, HFR_Opts opts)
{
  {
    const int reduce_times = opts.reduce_times;
    constexpr auto ConcatBlockDim = 128;
    using K = psz::HFR_PBK_Constants;
    buf->set_use_prebuilt_rvbk(true);
    const int magnitude = opts.magnitude;  // 10 = 1Ki (default), 11 = 2Ki, 12 = 4Ki
    const size_t pardeg = (len - 1) / ((size_t)1u << magnitude) + 1;
    const u4 stride_words = hfr_stride_words<E>(magnitude);
    auto launch_enc = [&]<int RT>(std::integral_constant<int, RT>, auto* hdrs) {
      if (magnitude >= 12) {
        using Enc4kA = phf::module::HFR_PBKC_encode<E, 12, RT, H4, K::Radius, /*IterLog=*/2>;
        // blockdim 256 doubles the threadblock (IterLog=1); r0 stays on 128 (2048 > launch cap).
        if constexpr (RT >= 1) {
          using Enc4kB = phf::module::HFR_PBKC_encode<E, 12, RT, H4, K::Radius, /*IterLog=*/1>;
          if (opts.blockdim >= 256) {
            Enc4kB::GPU_kernel(
                in, len, (H4*)pbk25_r128_book_d_ptr(), buf->bitstream_d(),
                (typename Enc4kB::header_t*)hdrs, opts.block_outliers, stream);
            return;
          }
        }
        Enc4kA::GPU_kernel(
            in, len, (H4*)pbk25_r128_book_d_ptr(), buf->bitstream_d(),
            (typename Enc4kA::header_t*)hdrs, opts.block_outliers, stream);
      }
      else if (magnitude >= 11) {
        using Enc2k = phf::module::HFR_PBKC_encode<E, 11, RT, H4, K::Radius, /*IterLog=*/1>;
        Enc2k::GPU_kernel(
            in, len, (H4*)pbk25_r128_book_d_ptr(), buf->bitstream_d(),
            (typename Enc2k::header_t*)hdrs, opts.block_outliers, stream);
      }
      else
        phf::module::HFR_PBKC_encode<E, K::Magnitude, RT, H4, K::Radius>::GPU_kernel(
            in, len, (H4*)pbk25_r128_book_d_ptr(), buf->bitstream_d(), hdrs, opts.block_outliers,
            stream);
    };
    auto launch_aggregate = [&]() {
      auto concat = [&]<int M>(std::integral_constant<int, M>) {
        using Concat = phf::_future_concat_via_scatter<E, ConcatBlockDim, M>;
        Concat::GPU_kernel(
            (typename Concat::bheader_t*)buf->pbk_headers_d(), buf->par_entry_d(),
            buf->bitstream_d(), buf->packed_d(), buf->pbk_packed_headers_d(), (u4)sizeof(H4),
            stride_words, (int)pardeg, buf->scan_partial_aggregate_d(), buf->scan_incl_prefix_d(),
            buf->scan_tile_status_d(), buf->total_ncell_d(), stream);
      };
      if (magnitude >= 12)
        concat(std::integral_constant<int, 12>{});
      else if (magnitude >= 11)
        concat(std::integral_constant<int, 11>{});
      else
        concat(std::integral_constant<int, 10>{});
    };
    switch (reduce_times) {
      case 0:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 0>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      case 1:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 1>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      case 2:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 2>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      case 3:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 3>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      case 4:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 4>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      default: return PHF_NOT_IMPLEMENTED;
    }
  }
}

template <typename E>
int encode_hfr_pbkgo(
    Buf<E>* buf, E* in, size_t const len, u1** out, size_t* outlen, phf_header& header,
    hf_stream_t stream, float* opt_ms_encoder, float* opt_ms_lago, HFR_Opts opts)
{
  {
    const int magnitude = opts.magnitude;
    const int reduce_times = opts.reduce_times;
    const size_t pardeg = (len - 1) / ((size_t)1u << magnitude) + 1;
    using K = psz::HFR_PBK_Constants;
    int rt = reduce_times;
    if (rt < 2) {
      fprintf(stderr, "[phf::warn] HFR-PBKGO falls back to rmerge-count >=2 (rt=%d).\n", rt);
      rt = 2;
    }
    buf->set_use_prebuilt_rvbk(true);
    buf->set_use_pbkgo(true);
    auto launch_enc = [&]<int RT>(std::integral_constant<int, RT>, auto* /*unused*/) {
      auto go = [&]<int M>(std::integral_constant<int, M>) {
        using Enc = phf::module::HFR_PBKGO_encode<E, M, RT, H4, K::Radius>;
        Enc::GPU_kernel(
            in, len, (H4*)pbk25_r128_book_d_ptr(), buf->bitstream_d(),
            (typename Enc::header_t*)buf->pbk_headers_d(), opts.block_outliers,
            buf->pbk_packed_headers_d(), buf->total_ncell_d(), buf->pbkgo_state_d(),
            buf->pbkgo_max_resident_blocks(), stream);
      };
      if (magnitude >= 12)
        go(std::integral_constant<int, 12>{});
      else if (magnitude >= 11)
        go(std::integral_constant<int, 11>{});
      else
        go(std::integral_constant<int, 10>{});
    };
    auto launch_aggregate = []() { /* no-op: encoder emitted everything */ };
    switch (rt) {
      case 2:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 2>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      case 3:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 3>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      case 4:
        return _HFR_common_enc(
            buf, len, out, outlen, header, stream,
            [&](auto* h) { launch_enc(std::integral_constant<int, 4>{}, h); }, launch_aggregate,
            opt_ms_encoder, opt_ms_lago, (int)pardeg);
      default: return PHF_NOT_IMPLEMENTED;
    }
  }
}

// Unified HFR-family decode: one GPU_kernel; variant picks (Storage, reverse-book source).
template <typename Ein, typename Eout = Ein>
int decode_hfr(
    Buf<Ein>* buf, phf_header& header, u1* in_encoded, Eout* out_decoded, hf_stream_t stream,
    psz_codec variant, int magnitude = 10)
{
  {
    using K = psz::HFR_PBK_Constants;
    constexpr auto RvbkBytesPerBook = (int)K::RvbkBytesPerBook;  // 512

    auto bs_ptr = (H4*)(in_encoded + header.entry[PHFHEADER_BITSTREAM]);
    auto packed_headers = (uint32_t const*)(in_encoded + header.entry[PHFHEADER_PBK_HEADERS]);
    const auto bs_bytes = header.total_ncell * sizeof(H4);
    const int pardeg = (int)header.pardeg;

    if (variant == HFR) {
      // runtime-built reverse book (Storage=Ein), one book for all blocks.
      auto rvbk = (u1*)(in_encoded + header.entry[PHFHEADER_RVBK]);
      const int rvbk_bytes =
          (int)(header.entry[PHFHEADER_RVBK + 1] - header.entry[PHFHEADER_RVBK]);
      if (magnitude >= 12)
        phf::module::HFR_PBK_decoder<Ein, H4, Ein, 12>::template GPU_kernel<Eout>(
            bs_ptr, bs_bytes, rvbk, rvbk_bytes, packed_headers, pardeg, header.ori_len,
            out_decoded, buf->incomp_flag_d(), stream);
      else if (magnitude >= 11)
        phf::module::HFR_PBK_decoder<Ein, H4, Ein, 11>::template GPU_kernel<Eout>(
            bs_ptr, bs_bytes, rvbk, rvbk_bytes, packed_headers, pardeg, header.ori_len,
            out_decoded, buf->incomp_flag_d(), stream);
      else
        phf::module::HFR_PBK_decoder<Ein, H4, Ein>::template GPU_kernel<Eout>(
            bs_ptr, bs_bytes, rvbk, rvbk_bytes, packed_headers, pardeg, header.ori_len,
            out_decoded, buf->incomp_flag_d(), stream);
    }
    else {
      // prebuilt pbk25_r128 pool (Storage=u1); V3/V4 offset to their single g_encid book.
      auto rvbk = (u1*)pbk25_r128_rvbk_d_ptr() + ((variant == HFR_V3 or variant == HFR_V4)
                                                      ? (size_t)header.g_encid * RvbkBytesPerBook
                                                      : 0);
      if (magnitude >= 12)
        phf::module::HFR_PBK_decoder<Ein, H4, u1, 12>::template GPU_kernel<Eout>(
            bs_ptr, bs_bytes, rvbk, RvbkBytesPerBook, packed_headers, pardeg, header.ori_len,
            out_decoded, buf->incomp_flag_d(), stream);
      else if (magnitude >= 11)
        phf::module::HFR_PBK_decoder<Ein, H4, u1, 11>::template GPU_kernel<Eout>(
            bs_ptr, bs_bytes, rvbk, RvbkBytesPerBook, packed_headers, pardeg, header.ori_len,
            out_decoded, buf->incomp_flag_d(), stream);
      else
        phf::module::HFR_PBK_decoder<Ein, H4, u1>::template GPU_kernel<Eout>(
            bs_ptr, bs_bytes, rvbk, RvbkBytesPerBook, packed_headers, pardeg, header.ori_len,
            out_decoded, buf->incomp_flag_d(), stream);
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

// HFR-v3 book source: pick one global PBK book from the histogram, on the GPU.
template <typename E>
int high_level<E>::HFR_pick_pbk(
    phf::Buf<E>* buf, u4* hist_d, u2 const bklen, size_t const len, hf_stream_t stream)
{
  buf->set_rt_bklen(psz::HFR_PBK_Constants::MaxDictsize);
  phf::module::HFR_pick_pbk(
      hist_d, (u4)bklen, len, (u4*)pbk25_r128_book_d_ptr(), buf->book_d(), buf->pick_encid_d(),
      stream);
  return 0;
}

template <typename E>
int high_level<E>::HF_encode(
    Buf<E>* buf, E* in, size_t const len, u1** out, size_t* outlen, phf_header& header,
    hf_stream_t stream, psz_codec variant, float* opt_ms_encoder, float* opt_ms_lago)
{
  switch (variant) {
    // HF is an alias for HFr2; the legacy SoA + ph3/ph4 host-scan path is retired.
    case HF:
    case HFr2:
      return dispatch::encode_hf_rev2<E>(
          buf, in, len, out, outlen, header, stream, opt_ms_encoder, opt_ms_lago);
    default: return PHF_NOT_IMPLEMENTED;
  }
}

template <typename E>
template <typename Eout>
int high_level<E>::HF_decode(
    Buf<E>* buf, phf_header& header, u1* in_encoded, Eout* out_decoded, hf_stream_t stream,
    psz_codec variant)
{
  // HF{,_r1}: same layout, so same decoder
  switch (variant) {
    case HF:
    case HFr2:
      return dispatch::decode_hf_rev2<E, Eout>(buf, header, in_encoded, out_decoded, stream);
    default: return PHF_NOT_IMPLEMENTED;
  }
}

template <typename E>
int high_level<E>::HFR_encode(
    Buf<E>* buf, E* in, size_t const len, u1** out, size_t* outlen, phf_header& header,
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
    case HFR_V4:
      return dispatch::encode_hfr_v4<E>(
          buf, in, len, out, outlen, header, stream, opt_ms_encoder, opt_ms_lago, opts);
    case HFR_PBKF: return PHF_NOT_IMPLEMENTED;
    default: return PHF_NOT_IMPLEMENTED;
  }
}

template <typename E>
template <typename Eout>
int high_level<E>::HFR_decode(
    Buf<E>* buf, phf_header& header, u1* in_encoded, Eout* out_decoded, hf_stream_t stream,
    psz_codec variant, int magnitude)
{
  switch (variant) {
    case HFR:
    case HFR_PBKC:
    case HFR_PBKGO:
    case HFR_V3:
    case HFR_V4:
      return dispatch::decode_hfr<E, Eout>(
          buf, header, in_encoded, out_decoded, stream, variant, magnitude);
    case HFR_PBKF: return PHF_NOT_IMPLEMENTED;
    default: return PHF_NOT_IMPLEMENTED;
  }
}

// FIXME: E and Eout inflate the binary
template <typename E>
template <typename Eout>
int high_level<E>::HFD26_decode(
    Buf<E>* buf, phf_header& header, u1* in_encoded, Eout* out_decoded, hf_stream_t stream,
    psz_codec variant, int magnitude)
{
  {
    const size_t pardeg = header.pardeg;
    auto bs_ptr = (H4*)(in_encoded + header.entry[PHFHEADER_BITSTREAM]);
    auto packed_headers = (u4 const*)(in_encoded + header.entry[PHFHEADER_PBK_HEADERS]);
    const size_t bs_bytes = header.total_ncell * sizeof(H4);

    auto lut_d = buf->lut_d();
    const bool build_lut = not buf->lut_ready();
    auto incomp_flag = buf->incomp_flag_d();

    auto launch = [&]<typename Storage>(u1* rvbk, int rvbk_bytes) {
      auto go = [&]<int Mag>(std::integral_constant<int, Mag>) {
        phf::module::HFD26<E, H4, Storage, Mag>::template decode_fused<Eout>(
            bs_ptr, bs_bytes, rvbk, rvbk_bytes, packed_headers, lut_d, (int)pardeg, header.ori_len,
            out_decoded, incomp_flag, stream);
      };
      if (magnitude >= 12)
        go(std::integral_constant<int, 12>{});
      else if (magnitude >= 11)
        go(std::integral_constant<int, 11>{});
      else
        go(std::integral_constant<int, 10>{});
    };

    switch (variant) {
      case HFR: {
        auto rvbk_ptr = (u1*)(in_encoded + header.entry[PHFHEADER_RVBK]);
        const int rvbk_bytes =
            (int)(header.entry[PHFHEADER_RVBK + 1] - header.entry[PHFHEADER_RVBK]);
        if (build_lut)
          phf::module::HFD26<E, H4, E>::build_lut(rvbk_ptr, rvbk_bytes, 1, lut_d, stream);
        launch.template operator()<E>(rvbk_ptr, rvbk_bytes);
        break;
      }
      case HFR_PBKC:
      case HFR_PBKGO: {
        constexpr auto RvbkBytesPerBook = psz::HFR_PBK_Constants::RvbkBytesPerBook;
        launch.template operator()<u1>((u1*)pbk25_r128_rvbk_d_ptr(), RvbkBytesPerBook);
        break;
      }
      case HFR_V3:
      case HFR_V4: {
        // ensure no silent/wrong-book decode
        constexpr auto RvbkBytesPerBook = psz::HFR_PBK_Constants::RvbkBytesPerBook;
        constexpr auto LutEntries = 256;
        auto rvbk_ptr = (u1*)pbk25_r128_rvbk_d_ptr() + (size_t)header.g_encid * RvbkBytesPerBook;
        lut_d = buf->lut_d() + (size_t)header.g_encid * LutEntries;
        launch.template operator()<u1>(rvbk_ptr, RvbkBytesPerBook);
        break;
      }
      default: return PHF_NOT_IMPLEMENTED;
    }
    if (build_lut) buf->lut_ready(true);
    sync_by_stream(stream);
    return 0;
  }
}

}  // namespace phf

template struct phf::high_level<u1>;
template struct phf::high_level<u2>;
template struct phf::high_level<u4>;

template int phf::high_level<u1>::HFR_decode<u1>(
    phf::Buf<u1>*, phf_header&, u1*, u1*, hf_stream_t, psz_codec, int);

// HFD26: 1- and 2-byte Eq decode; the u4 cells resolve to the stub.
template int phf::high_level<u1>::HFD26_decode<u1>(
    phf::Buf<u1>*, phf_header&, u1*, u1*, hf_stream_t, psz_codec, int);
template int phf::high_level<u2>::HFD26_decode<u2>(
    phf::Buf<u2>*, phf_header&, u1*, u2*, hf_stream_t, psz_codec, int);
template int phf::high_level<u2>::HFD26_decode<f4>(
    phf::Buf<u2>*, phf_header&, u1*, f4*, hf_stream_t, psz_codec, int);
template int phf::high_level<u2>::HFD26_decode<f8>(
    phf::Buf<u2>*, phf_header&, u1*, f8*, hf_stream_t, psz_codec, int);
template int phf::high_level<u4>::HFD26_decode<u4>(
    phf::Buf<u4>*, phf_header&, u1*, u4*, hf_stream_t, psz_codec, int);
template int phf::high_level<u4>::HFD26_decode<f4>(
    phf::Buf<u4>*, phf_header&, u1*, f4*, hf_stream_t, psz_codec, int);
template int phf::high_level<u4>::HFD26_decode<f8>(
    phf::Buf<u4>*, phf_header&, u1*, f8*, hf_stream_t, psz_codec, int);

template int phf::high_level<u1>::HF_decode<u1>(
    phf::Buf<u1>*, phf_header&, u1*, u1*, hf_stream_t, psz_codec);
template int phf::high_level<u2>::HF_decode<u2>(
    phf::Buf<u2>*, phf_header&, u1*, u2*, hf_stream_t, psz_codec);
template int phf::high_level<u2>::HFR_decode<u2>(
    phf::Buf<u2>*, phf_header&, u1*, u2*, hf_stream_t, psz_codec, int);

template int phf::high_level<u2>::HF_decode<f4>(
    phf::Buf<u2>*, phf_header&, u1*, f4*, hf_stream_t, psz_codec);
template int phf::high_level<u2>::HF_decode<f8>(
    phf::Buf<u2>*, phf_header&, u1*, f8*, hf_stream_t, psz_codec);
template int phf::high_level<u2>::HFR_decode<f4>(
    phf::Buf<u2>*, phf_header&, u1*, f4*, hf_stream_t, psz_codec, int);
template int phf::high_level<u2>::HFR_decode<f8>(
    phf::Buf<u2>*, phf_header&, u1*, f8*, hf_stream_t, psz_codec, int);

template int phf::high_level<u4>::HF_decode<u4>(
    phf::Buf<u4>*, phf_header&, u1*, u4*, hf_stream_t, psz_codec);
template int phf::high_level<u4>::HFR_decode<u4>(
    phf::Buf<u4>*, phf_header&, u1*, u4*, hf_stream_t, psz_codec, int);

template int phf::high_level<u4>::HF_decode<f4>(
    phf::Buf<u4>*, phf_header&, u1*, f4*, hf_stream_t, psz_codec);
template int phf::high_level<u4>::HF_decode<f8>(
    phf::Buf<u4>*, phf_header&, u1*, f8*, hf_stream_t, psz_codec);
template int phf::high_level<u4>::HFR_decode<f4>(
    phf::Buf<u4>*, phf_header&, u1*, f4*, hf_stream_t, psz_codec, int);
template int phf::high_level<u4>::HFR_decode<f8>(
    phf::Buf<u4>*, phf_header&, u1*, f8*, hf_stream_t, psz_codec, int);
