// (2026-05-17) Synced from bleeding-edge @ 93f248bb.
// (2026-06-17) The decoded is type Eout so that HFR_* can write eq directly to f4/f8 output buffer.

#include <cstdint>

#include "hfr-pbk.hh"
#include "hfr-pbk_decoder.hh"
#include "single_inflate.inl"

namespace phf {

__forceinline__ __device__ u4 unpack_par_nbit(u4 w0) { return w0 >> 14; }
__forceinline__ __device__ u4 unpack_par_encid(u4 w0) { return (w0 >> 9) & 0x1Fu; }
template <typename H>
__forceinline__ __device__ u4 unpack_par_entry_words(u4 w1)
{
  return w1 / (u4)sizeof(H);
}

template <typename Ein, typename H, typename Storage, typename Eout = Ein>
__global__ void KCU_HFR_PBK_decode(
    H* in_pbk_bitstream, size_t const pbk_bitstream_len, u1* in_rvbk_r128_25, int const rvbk_nbyte,
    u4 const* pbk_packed_headers, int const pbk_pardeg, size_t const data_len, Eout* out_decoded)
{
  using BreakCell = psz::HFR_PBK_Breaks<psz::HFR_PBK_Constants::Radius>;
  constexpr auto ChunkSize = 1024;
  constexpr auto NumBooks = (int)psz::HFR_PBK_Constants::NumBooks;
  auto gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= pbk_pardeg) return;

  // valid count for this block; the last block is partial when data_len % ChunkSize != 0.
  size_t const block_off = (size_t)ChunkSize * gid;
  u4 const valid = (u4)(data_len - block_off < (size_t)ChunkSize ? data_len - block_off : ChunkSize);

  u4 const w0 = pbk_packed_headers[2 * gid + 0];
  u4 const w1 = pbk_packed_headers[2 * gid + 1];
  u4 const tree_idx = unpack_par_encid(w0);
  u4 const unit_start = unpack_par_entry_words<H>(w1);

  // Pass-through fallback: enc_id >= NumBooks -> raw E[ChunkSize] in the slot.
  if (tree_idx >= (u4)NumBooks) {
    auto raw = reinterpret_cast<Ein*>(in_pbk_bitstream + unit_start);
    auto dst = out_decoded + block_off;
    for (u4 i = 0; i < valid; i++) dst[i] = (Eout)raw[i];
    return;
  }

  // Recover n_breaks from par_entry delta; per-block layout [breaks | bitstream].
  u4 const bit_count = unpack_par_nbit(w0);
  u4 const total_words =
      (gid + 1 < pbk_pardeg)
          ? (unpack_par_entry_words<H>(pbk_packed_headers[2 * (gid + 1) + 1]) - unit_start)
          : ((u4)(pbk_bitstream_len / sizeof(H)) - unit_start);
  u4 const bs_words = (bit_count + 31) / 32;
  u4 const n_breaks = total_words - bs_words;
  auto const block_slot = in_pbk_bitstream + unit_start;
  auto const br_slot = reinterpret_cast<BreakCell const*>(block_slot);
  auto const bs_slot = block_slot + n_breaks;

  auto rvbk = in_rvbk_r128_25 + tree_idx * rvbk_nbyte;
  auto out_block = out_decoded + block_off;
  phf::single_thread_inflate<Ein, H, Storage, Eout>(
      bs_slot, out_block, rvbk, (int)bit_count, (int)valid);

  for (u4 k = 0; k < n_breaks; k++) {
    auto cell = br_slot[k];
    if (cell.idx < valid) out_block[cell.idx] = (Eout)cell.val;
  }
}

}  // namespace phf

namespace phf::module {

template <typename E, typename H, typename Storage>
template <typename Eout>
int HFR_PBK_decoder<E, H, Storage>::GPU_kernel(
    H* in_pbk_bitstream, size_t pbk_bitstream_len, u1* in_rvbk_r128_25, int rvbk_nbyte,
    u4 const* pbk_packed_headers, int pbk_pardeg, size_t data_len, Eout* out_decoded, void* stream)
{
  if (pbk_pardeg <= 0) return 0;
  constexpr int BlockDim = 128;
  dim3 grid((unsigned)((pbk_pardeg + BlockDim - 1) / BlockDim), 1, 1);
  dim3 block(BlockDim, 1, 1);
  phf::KCU_HFR_PBK_decode<E, H, Storage, Eout><<<grid, block, 0, (cudaStream_t)stream>>>(
      in_pbk_bitstream, pbk_bitstream_len, in_rvbk_r128_25, rvbk_nbyte, pbk_packed_headers,
      pbk_pardeg, data_len, out_decoded);
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
    u4*, size_t, u1*, int, u4 const*, int, size_t, u2*, void*);
template int HFR_PBK_decoder<u2, u4, u2>::GPU_kernel<u2>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, u2*, void*);

template int HFR_PBK_decoder<u2, u4, u1>::GPU_kernel<f4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f4*, void*);
template int HFR_PBK_decoder<u2, u4, u1>::GPU_kernel<f8>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f8*, void*);
template int HFR_PBK_decoder<u2, u4, u2>::GPU_kernel<f4>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f4*, void*);
template int HFR_PBK_decoder<u2, u4, u2>::GPU_kernel<f8>(
    u4*, size_t, u1*, int, u4 const*, int, size_t, f8*, void*);

}  // namespace phf::module
