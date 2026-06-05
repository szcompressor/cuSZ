
#include <cstdio>

#include "hfr-pbk.hh"  // psz::HFR_PBK_Breaks<128>
#include "auxiliary.inl"
#include "hf_impl.hh"
#include "hfr.hh"

using u4 = uint32_t;

#define RETURN_AT(BreakPoint) \
  if constexpr (return_at == BreakPoint) return;

namespace phf {

namespace {
static const int HFReVISIT_disable_trap = 0;
}

template <class C>
__forceinline__ __device__ void hfr_data_load(
    typename C::T* in, u4 inlen, typename C::Hf* dram_book, u4 bklen,
    volatile typename C::T* s_to_encode, volatile typename C::Hf* s_book)
{
  auto entry = C::ChunkSize * blockIdx.x;
  auto allowed_len = min((u4)C::ChunkSize, inlen - entry);

  for (auto i = threadIdx.x; i < bklen; i += blockDim.x) s_book[i] = dram_book[i];

  // Pad last block with neutral (center) symbol so book lookup is valid.
  for (auto i = threadIdx.x; i < C::ChunkSize; i += blockDim.x)
    s_to_encode[i] = (i < allowed_len) ? in[entry + i] : (typename C::T)(bklen / 2);
  __syncthreads();
}

// Original RS-merge design.
template <class C, typename BreakCell>
__forceinline__ __device__ void reduce_merge_reference(
    u4 inlen, typename C::Hf alt_code, u4 alt_bitcount, volatile typename C::T* s_to_encode,
    volatile typename C::Hf* s_book, volatile typename C::Hf* s_reduced, volatile u4* s_bitcount,
    BreakCell* s_breaks, u4* s_brnum)
{
  using Hf = typename C::Hf;
  using W = typename C::W;
  using u2 = uint16_t;

  auto bits_of = [](Hf* _w) { return reinterpret_cast<W*>(_w)->bitcount; };

  auto entry = (u4)(C::ChunkSize * blockIdx.x);
  auto allowed_len = min((u4)C::ChunkSize, inlen - entry);

  constexpr auto PerSymBudget = C::BITWIDTH / C::ShardSize;

  auto p_bits{0u};
  Hf p_reduced{0x0};

  for (auto i = 0; i < C::ShardSize; i++) {
    auto idx = (threadIdx.x * C::ShardSize) + i;
    auto p_key = (Hf)s_to_encode[idx];
    auto p_val = (Hf)s_book[p_key];
    auto sym_bits = bits_of(&p_val);

    p_val <<= (C::BITWIDTH - sym_bits);
    p_reduced |= (p_val >> p_bits);
    p_bits += sym_bits * (idx < allowed_len);
  }

  if (p_bits > C::BITWIDTH) {
    if (alt_bitcount == 0) { atomicAdd(s_brnum, (u4)C::NumShards + 1); }
    else {
      p_bits = 0u;
      p_reduced = 0x0;

      for (auto i = 0; i < C::ShardSize; i++) {
        auto idx = (threadIdx.x * C::ShardSize) + i;
        auto p_key = (Hf)s_to_encode[idx];
        auto p_val = (Hf)s_book[p_key];
        auto sym_bits = bits_of(&p_val);

        if (sym_bits > PerSymBudget && idx < allowed_len) {
          auto pos = atomicAdd(s_brnum, 1u);
          if (pos < (u4)C::NumShards) { s_breaks[pos] = {(u2)p_key, (u2)idx}; }
          p_val = alt_code;
          sym_bits = alt_bitcount;
        }
        p_val <<= (C::BITWIDTH - sym_bits);
        p_reduced |= (p_val >> p_bits);
        p_bits += sym_bits * (idx < allowed_len);
      }
    }
  }

  s_reduced[threadIdx.x] = p_reduced;
  s_bitcount[threadIdx.x] = p_bits;
  __syncthreads();
}

template <class C>
__forceinline__ __device__ void shuffle_merge_reference(
    volatile typename C::Hf* s_reduced, volatile u4* s_bitcount)
{
  using Hf = typename C::Hf;

  for (auto sf = C::ShuffleTimes, stride = 1u; sf > 0; sf--, stride *= 2) {
    const auto l = threadIdx.x & ~(stride * 2 - 1);
    const auto r = l + stride;

    const auto lbc = s_bitcount[l];
    const u4 used__units = lbc / C::BITWIDTH;
    const u4 used___bits = lbc & (C::BITWIDTH - 1);
    const u4 unused_bits = C::BITWIDTH - used___bits;

    auto lend = s_reduced + l + used__units;
    auto this_point = s_reduced[threadIdx.x];
    auto lsym = this_point >> used___bits;
    auto rsym = this_point << unused_bits;

    if (threadIdx.x >= r and threadIdx.x < r + stride)
      atomicAnd(const_cast<Hf*>(s_reduced + threadIdx.x), 0x0u);
    __syncthreads();

    if (threadIdx.x >= r and threadIdx.x < r + stride) {
      atomicOr(const_cast<Hf*>(lend + threadIdx.x - r + 0), lsym);
      atomicOr(const_cast<Hf*>(lend + threadIdx.x - r + 1), rsym);
    }

    if (threadIdx.x == l) s_bitcount[l] += s_bitcount[r];
    __syncthreads();
  }
}

template <typename BreakCell>
__forceinline__ __device__ void hfr_drain_breaks(
    BreakCell* s_breaks, u4 s_brnum, BreakCell* sp_breaks, u4* sp_count, u4* par_brnum,
    u4* par_broffset)
{
  __shared__ u4 s_offset;
  if (threadIdx.x == 0) {
    s_offset = atomicAdd(sp_count, s_brnum);
    par_brnum[blockIdx.x] = s_brnum;
    par_broffset[blockIdx.x] = s_offset;
  }
  __syncthreads();

  for (auto i = threadIdx.x; i < s_brnum; i += blockDim.x) sp_breaks[s_offset + i] = s_breaks[i];
}

template <class C>
__forceinline__ __device__ void hfr_data_store(
    volatile typename C::Hf* s_reduced, volatile u4* s_bitcount, typename C::Hf* dn_bitstream,
    u4* dn_bitcount)
{
  const auto bc_this_block = s_bitcount[0];
  const auto n_cell = (bc_this_block + C::BITWIDTH - 1) / C::BITWIDTH;

  if (threadIdx.x < n_cell)
    dn_bitstream[C::ChunkSize * blockIdx.x + threadIdx.x] = (typename C::Hf)s_reduced[threadIdx.x];

  if (threadIdx.x == 0) dn_bitcount[blockIdx.x] = bc_this_block;
}

template <class C>
__forceinline__ __device__ void hfr_handle_incomp(
    typename C::T* in, u4 inlen, typename C::Hf* dn_bitstream, u4* dn_bitcount)
{
  using T = typename C::T;
  constexpr auto ChunkSize = C::ChunkSize;
  auto entry = (u4)(ChunkSize * blockIdx.x);
  auto allowed_len = min((u4)ChunkSize, inlen - entry);

  // Write raw T values; past-end padded with 0 for deterministic cells.
  auto base = (T*)(dn_bitstream + ChunkSize * blockIdx.x);
  for (auto i = threadIdx.x; i < ChunkSize; i += blockDim.x)
    base[i] = (i < allowed_len) ? in[entry + i] : (T)0;

  if (threadIdx.x == 0) dn_bitcount[blockIdx.x] = ChunkSize * sizeof(T) * 8u;
}

template <class C, int return_at = HFReVISIT_disable_trap>
__global__ void KCU_HFReVISIT_encode(
    typename C::T* in, u4 inlen, typename C::Hf* dram_book, u4 bklen, typename C::Hf alt_code,
    u4 alt_bitcount, typename C::Hf* dn_bitstream, u4* dn_bitcount,
    psz::HFR_PBK_Breaks<128>* sp_breaks, u4* sp_count, u4* par_brnum, u4* par_broffset,
    uint8_t* par_encid)
{
  static_assert(C::ReduceTimes >= 1, "ReduceTimes must be >= 1.");
  static_assert((2 << C::Magnitude) < 98304, "Shared memory exceeds limit.");

  using BreakCell = psz::HFR_PBK_Breaks<128>;

  __shared__ typename C::T s_to_encode[C::ChunkSize];
  __shared__ typename C::Hf s_book[1024];
  __shared__ typename C::Hf s_reduced[C::NumShards];
  __shared__ u4 s_bitcount[C::NumShards + 1];
  __shared__ BreakCell s_breaks[C::NumShards];
  __shared__ u4 s_brnum;

  if (threadIdx.x == 0) s_brnum = 0;

  hfr_data_load<C>(in, inlen, dram_book, bklen, s_to_encode, s_book);
  RETURN_AT(1);

  reduce_merge_reference<C>(
      inlen, alt_code, alt_bitcount, s_to_encode, s_book, s_reduced, s_bitcount, s_breaks,
      &s_brnum);
  RETURN_AT(2);

  __shared__ int s_is_incomp;
  if (threadIdx.x == 0) s_is_incomp = (s_brnum > (u4)C::NumShards) ? 1 : 0;
  __syncthreads();

  if (s_is_incomp) {
    hfr_handle_incomp<C>(in, inlen, dn_bitstream, dn_bitcount);
    if (threadIdx.x == 0) {
      par_encid[blockIdx.x] = 1;
      par_brnum[blockIdx.x] = 0;
      par_broffset[blockIdx.x] = 0;
    }
    return;
  }

  shuffle_merge_reference<C>(s_reduced, s_bitcount);
  RETURN_AT(3);

  hfr_data_store<C>(s_reduced, s_bitcount, dn_bitstream, dn_bitcount);
  hfr_drain_breaks<BreakCell>(s_breaks, s_brnum, sp_breaks, sp_count, par_brnum, par_broffset);
  if (threadIdx.x == 0) par_encid[blockIdx.x] = 0;
}

}  // namespace phf

namespace phf::module {

template <typename T, int Magnitude, int ReduceTimes, bool UseScan, typename Hf>
int HFR_encoder<T, Magnitude, ReduceTimes, UseScan, Hf>::GPU_kernel_v1(
    T* in, size_t len, Hf* bk, u4 bklen, Hf alt_code, u4 alt_bitcount, Hf* dn_bitstream,
    u4* dn_bitcount, psz::HFR_PBK_Breaks<128>* sp_breaks, u4* sp_count, u4* par_brnum,
    u4* par_broffset, uint8_t* par_encid, void* stream)
{
  using C = HFR_Config<T, Magnitude, ReduceTimes, Hf>;

  constexpr auto nthread = C::BlockDim;
  const auto nblock = (len - 1) / C::ChunkSize + 1;

  phf::KCU_HFReVISIT_encode<C><<<nblock, nthread, 0, (cudaStream_t)stream>>>(
      in, (u4)len, bk, bklen, alt_code, alt_bitcount, dn_bitstream, dn_bitcount, sp_breaks,
      sp_count, par_brnum, par_broffset, par_encid);

  return 0;
}

}  // namespace phf::module

#define __INSTANTIATE_HFR(T, MAG, RED, SCAN) \
  template struct phf::module::HFR_encoder<T, MAG, RED, SCAN, u4>;

#define __INSTANTIATE_HFR_TYPES(MAG, RED, SCAN) \
  __INSTANTIATE_HFR(u4, MAG, RED, SCAN)         \
  __INSTANTIATE_HFR(u2, MAG, RED, SCAN)         \
  __INSTANTIATE_HFR(u1, MAG, RED, SCAN)

#define __INSTANTIATE_HFR_MAGS(RED, SCAN) \
  __INSTANTIATE_HFR_TYPES(12, RED, SCAN)  \
  __INSTANTIATE_HFR_TYPES(11, RED, SCAN)  \
  __INSTANTIATE_HFR_TYPES(10, RED, SCAN)  \
  __INSTANTIATE_HFR_TYPES(9, RED, SCAN)   \
  __INSTANTIATE_HFR_TYPES(8, RED, SCAN)   \
  __INSTANTIATE_HFR_TYPES(7, RED, SCAN)   \
  __INSTANTIATE_HFR_TYPES(6, RED, SCAN)   \
  __INSTANTIATE_HFR_TYPES(5, RED, SCAN)

#define __INSTANTIATE_RSMERGE_1(RED) \
  __INSTANTIATE_HFR_MAGS(RED, false) \
  __INSTANTIATE_HFR_MAGS(RED, true)
