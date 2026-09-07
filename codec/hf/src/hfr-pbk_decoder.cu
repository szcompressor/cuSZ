// (2026-05-17) Synced from bleeding-edge @ 93f248bb.
// (2026-06-17) The decoded is type Eout so that HFR_* can write eq directly to f4/f8 output buffer.

#include <cstdint>

#include "hfr-pbk.hh"
#include "hfr-pbk_decoder.hh"
#include "single_inflate.inl"

namespace phf {

using psz::unpack_par_dense;
using psz::unpack_par_encid;
using psz::unpack_par_end_words;
using psz::unpack_par_entry_words;
using psz::unpack_par_nunpred;

template <typename Ein, typename H, typename Storage, typename Eout = Ein, int Magnitude = 10>
__global__ void KCU_HFR_PBK_decode(
    H* in_pbk_bitstream, size_t const pbk_bitstream_len, u1* in_rvbk_r128_25, int const rvbk_nbyte,
    u4 const* pbk_packed_headers, int const pbk_pardeg, size_t const data_len, Eout* out_decoded,
    u1* out_incomp_flag)
{
  using BreakCell = psz::HFR_PBK_Breaks<psz::HFR_PBK_Constants::Radius>;
  constexpr auto ChunkSize = 1 << Magnitude;
  constexpr auto NumBooks = (int)psz::HFR_PBK_Constants::NumBooks;
  auto gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= pbk_pardeg) return;

  // valid count for this block (padding fine)
  size_t const block_off = (size_t)ChunkSize * gid;
  u4 const valid = (u4)(data_len - block_off < (size_t)ChunkSize ? data_len - block_off : ChunkSize);

  u4 const w0 = pbk_packed_headers[2 * gid + 0];
  u4 const tree_idx = unpack_par_encid<Magnitude>(w0);
  u4 const unit_start = unpack_par_entry_words<H>(pbk_packed_headers, gid);

  using psz::OutlierCell;
  bool const is_incomp_unpred = (tree_idx == (u4)psz::HFR_PBK_Constants::CodeIncompUnpred);
  if (out_incomp_flag) out_incomp_flag[gid] = is_incomp_unpred ? 1u : 0u;

  // pass-through: copy raw value
  if (tree_idx >= (u4)NumBooks) {
    auto dst = out_decoded + block_off;
    if (is_incomp_unpred) {
      auto raw = reinterpret_cast<Ein const*>(in_pbk_bitstream + unit_start);
      for (u4 i = 0; i < valid; i++) dst[i] = (Eout)psz::incomp_unpack<Ein>(raw[i]);
    }
    else {  // breaks-incomp (30)
      auto raw = reinterpret_cast<Ein*>(in_pbk_bitstream + unit_start);
      for (u4 i = 0; i < valid; i++) dst[i] = (Eout)raw[i];
      u4 const n_unpred = unpack_par_nunpred<Magnitude>(w0);
      u4 const content_words = unpack_par_dense<Magnitude>(w0);  // raw eq words
      auto cells =
          reinterpret_cast<OutlierCell const*>(in_pbk_bitstream + unit_start + content_words);
      for (u4 k = 0; k < n_unpred; k++) {
        auto cell = cells[k];
        if (cell.idx < valid) dst[cell.idx] = (Eout)cell.val;
      }
    }
    return;
  }

  // per-block layout [breaks | bitstream | unpred].
  u4 const bs_words = unpack_par_dense<Magnitude>(w0);
  u4 const n_unpred = unpack_par_nunpred<Magnitude>(w0);
  u4 const total_words =
      unpack_par_end_words<H>(pbk_packed_headers, gid, pbk_pardeg, pbk_bitstream_len) - unit_start;
  // (matche write_pbk_bitstream_v2).
  u4 const unpred_words = ((n_unpred * (u4)sizeof(OutlierCell) + 3u) & ~3u) / (u4)sizeof(H);
  u4 const n_breaks = total_words - bs_words - unpred_words;
  auto const block_slot = in_pbk_bitstream + unit_start;
  auto const br_slot = reinterpret_cast<BreakCell const*>(block_slot);
  auto const bs_slot = block_slot + n_breaks;

  auto rvbk = in_rvbk_r128_25 + tree_idx * rvbk_nbyte;
  auto out_block = out_decoded + block_off;
  phf::single_thread_inflate<Ein, H, Storage, Eout>(
      bs_slot, out_block, rvbk, (int)(bs_words * 32u), (int)valid);

  for (u4 k = 0; k < n_breaks; k++) {
    auto cell = br_slot[k];
    if (cell.idx < valid) out_block[cell.idx] = (Eout)cell.val;
  }

  // per-block outliers to replace eq
  auto const up_slot = reinterpret_cast<OutlierCell const*>(bs_slot + bs_words);
  for (u4 k = 0; k < n_unpred; k++) {
    auto cell = up_slot[k];
    if (cell.idx < valid) out_block[cell.idx] = (Eout)cell.val;
  }
}

}  // namespace phf

namespace phf::module {

template <typename E, typename H, typename Storage, int Magnitude>
template <typename Eout>
int HFR_PBK_decoder<E, H, Storage, Magnitude>::GPU_kernel(
    H* in_pbk_bitstream, size_t pbk_bitstream_len, u1* in_rvbk_r128_25, int rvbk_nbyte,
    u4 const* pbk_packed_headers, int pbk_pardeg, size_t data_len, Eout* out_decoded,
    u1* out_incomp_flag, void* stream)
{
  if (pbk_pardeg <= 0) return 0;
  constexpr int BlockDim = 128;
  dim3 grid((unsigned)((pbk_pardeg + BlockDim - 1) / BlockDim), 1, 1);
  dim3 block(BlockDim, 1, 1);
  phf::KCU_HFR_PBK_decode<E, H, Storage, Eout, Magnitude><<<grid, block, 0, (cudaStream_t)stream>>>(
      in_pbk_bitstream, pbk_bitstream_len, in_rvbk_r128_25, rvbk_nbyte, pbk_packed_headers,
      pbk_pardeg, data_len, out_decoded, out_incomp_flag);
  return 0;
}

// PBKC (u1-keyed prebuilt rvbk).
template struct HFR_PBK_decoder<u1, u4, u1>;
template struct HFR_PBK_decoder<u2, u4, u1>;
template struct HFR_PBK_decoder<u4, u4, u1>;
// HFR (runtime rvbk; Storage = E).
template struct HFR_PBK_decoder<u2, u4, u2>;
template struct HFR_PBK_decoder<u4, u4, u4>;

template int HFR_PBK_decoder<u2, u4, u1>::GPU_kernel<u2>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, u2*, u1*, void*);
template int HFR_PBK_decoder<u2, u4, u2>::GPU_kernel<u2>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, u2*, u1*, void*);

template int HFR_PBK_decoder<u2, u4, u1>::GPU_kernel<f4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f4*, u1*, void*);
template int HFR_PBK_decoder<u2, u4, u1>::GPU_kernel<f8>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f8*, u1*, void*);
template int HFR_PBK_decoder<u2, u4, u2>::GPU_kernel<f4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f4*, u1*, void*);
template int HFR_PBK_decoder<u2, u4, u2>::GPU_kernel<f8>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f8*, u1*, void*);

template int HFR_PBK_decoder<u4, u4, u1>::GPU_kernel<u4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, u4*, u1*, void*);
template int HFR_PBK_decoder<u4, u4, u1>::GPU_kernel<f4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f4*, u1*, void*);
template int HFR_PBK_decoder<u4, u4, u1>::GPU_kernel<f8>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f8*, u1*, void*);
template int HFR_PBK_decoder<u4, u4, u4>::GPU_kernel<u4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, u4*, u1*, void*);
template int HFR_PBK_decoder<u4, u4, u4>::GPU_kernel<f4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f4*, u1*, void*);
template int HFR_PBK_decoder<u4, u4, u4>::GPU_kernel<f8>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f8*, u1*, void*);

// u1 byte-symbol profile (struct is instantiated in the PBKC block above).
template int HFR_PBK_decoder<u1, u4, u1>::GPU_kernel<u1>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, u1*, u1*, void*);

// 2Ki (Magnitude=11): PBKC round-trip (bin_hf uses Eout=u2; f4/f8 for the pipeline).
template struct HFR_PBK_decoder<u2, u4, u1, 11>;
template int HFR_PBK_decoder<u2, u4, u1, 11>::GPU_kernel<u2>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, u2*, u1*, void*);
template int HFR_PBK_decoder<u2, u4, u1, 11>::GPU_kernel<f4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f4*, u1*, void*);
template int HFR_PBK_decoder<u2, u4, u1, 11>::GPU_kernel<f8>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f8*, u1*, void*);

// 4Ki (Magnitude=12): same shape as 2Ki.
template struct HFR_PBK_decoder<u2, u4, u1, 12>;
template int HFR_PBK_decoder<u2, u4, u1, 12>::GPU_kernel<u2>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, u2*, u1*, void*);
template int HFR_PBK_decoder<u2, u4, u1, 12>::GPU_kernel<f4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f4*, u1*, void*);
template int HFR_PBK_decoder<u2, u4, u1, 12>::GPU_kernel<f8>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f8*, u1*, void*);

// PBKC round-trip at Ein=u4, 2Ki / 4Ki.
template struct HFR_PBK_decoder<u4, u4, u1, 11>;
template int HFR_PBK_decoder<u4, u4, u1, 11>::GPU_kernel<u4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, u4*, u1*, void*);
template int HFR_PBK_decoder<u4, u4, u1, 11>::GPU_kernel<f4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f4*, u1*, void*);
template int HFR_PBK_decoder<u4, u4, u1, 11>::GPU_kernel<f8>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f8*, u1*, void*);
template struct HFR_PBK_decoder<u4, u4, u1, 12>;
template int HFR_PBK_decoder<u4, u4, u1, 12>::GPU_kernel<u4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, u4*, u1*, void*);
template int HFR_PBK_decoder<u4, u4, u1, 12>::GPU_kernel<f4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f4*, u1*, void*);
template int HFR_PBK_decoder<u4, u4, u1, 12>::GPU_kernel<f8>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f8*, u1*, void*);

// u1 byte-symbol profile at 2Ki / 4Ki.
template struct HFR_PBK_decoder<u1, u4, u1, 11>;
template int HFR_PBK_decoder<u1, u4, u1, 11>::GPU_kernel<u1>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, u1*, u1*, void*);
template struct HFR_PBK_decoder<u1, u4, u1, 12>;
template int HFR_PBK_decoder<u1, u4, u1, 12>::GPU_kernel<u1>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, u1*, u1*, void*);

// HFR (runtime rvbk; Storage = E) at 2Ki / 4Ki.
template struct HFR_PBK_decoder<u2, u4, u2, 11>;
template int HFR_PBK_decoder<u2, u4, u2, 11>::GPU_kernel<u2>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, u2*, u1*, void*);
template int HFR_PBK_decoder<u2, u4, u2, 11>::GPU_kernel<f4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f4*, u1*, void*);
template int HFR_PBK_decoder<u2, u4, u2, 11>::GPU_kernel<f8>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f8*, u1*, void*);
template struct HFR_PBK_decoder<u2, u4, u2, 12>;
template int HFR_PBK_decoder<u2, u4, u2, 12>::GPU_kernel<u2>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, u2*, u1*, void*);
template int HFR_PBK_decoder<u2, u4, u2, 12>::GPU_kernel<f4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f4*, u1*, void*);
template int HFR_PBK_decoder<u2, u4, u2, 12>::GPU_kernel<f8>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f8*, u1*, void*);

// HFR (runtime rvbk; Storage = E) at Ein=u4, 2Ki / 4Ki.
template struct HFR_PBK_decoder<u4, u4, u4, 11>;
template int HFR_PBK_decoder<u4, u4, u4, 11>::GPU_kernel<u4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, u4*, u1*, void*);
template int HFR_PBK_decoder<u4, u4, u4, 11>::GPU_kernel<f4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f4*, u1*, void*);
template int HFR_PBK_decoder<u4, u4, u4, 11>::GPU_kernel<f8>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f8*, u1*, void*);
template struct HFR_PBK_decoder<u4, u4, u4, 12>;
template int HFR_PBK_decoder<u4, u4, u4, 12>::GPU_kernel<u4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, u4*, u1*, void*);
template int HFR_PBK_decoder<u4, u4, u4, 12>::GPU_kernel<f4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f4*, u1*, void*);
template int HFR_PBK_decoder<u4, u4, u4, 12>::GPU_kernel<f8>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f8*, u1*, void*);

}  // namespace phf::module
