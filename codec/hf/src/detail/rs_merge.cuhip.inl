/**
 * @file rs_merge.cuhip.inl
 * @author Jiannan Tian
 * @brief HF-ReVISIT: reduce-shuffle-merge Huffman encoding kernel.
 * @version 0.2
 * @date 2020-05-30 (created); 2025-10 (revised for psz_cusz merge-HFR)
 *
 * (C) 2020 by Washington State University, Argonne National Laboratory
 *
 * Differences from EIP's eip_c.cuhip.inl:
 *  - No blockwise header or prebuilt codebook; uses a single dynamic codebook.
 *  - No random-access archiving; dense output is at a fixed sequential offset
 *    per block (C::ChunkSize * blockIdx.x).
 *  - Breaking points are gathered into a sparse (val, idx) buffer and replaced
 *    inline by alt_code (the most-frequent-symbol codeword packed ReduceTimes).
 *    The decoder scatters them back from that buffer.
 *  - No FP16 incompressible fallback.
 */

#include <cstdio>

#include "auxiliary.inl"
#include "hf_impl.hh"
#include "rs_merge.hh"

using u4 = uint32_t;

#define RETURN_AT(BreakPoint) \
  if constexpr (return_at == BreakPoint) return;

namespace phf {

namespace {
static const int HFReVISIT_disable_trap = 0;
}

// ---------------------------------------------------------------------------
// Phase 1: load codebook + data into shared memory
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Phase 2: reduce-merge (per-thread shard) + breaking-point compaction
//
// Each thread encodes C::ShardSize consecutive symbols into a single Hf word
// (left-aligned accumulation from MSB).  If the accumulated bit count exceeds
// C::BITWIDTH, all ShardSize originals are spilled to the sparse buffer and the
// shard slot receives alt_code (= most-frequent symbol packed ReduceTimes).
// ---------------------------------------------------------------------------
template <class C>
__forceinline__ __device__ void hfr_reduce_merge_compact(
    u4 inlen, typename C::Hf alt_code, u4 alt_bitcount, volatile typename C::T* s_to_encode,
    volatile typename C::Hf* s_book, volatile typename C::Hf* s_reduced, volatile u4* s_bitcount,
    typename C::T* sp_val, u4* sp_idx, u4* sp_count)
{
  using Hf = typename C::Hf;
  using W = typename C::W;

  auto bits_of = [](Hf* _w) { return reinterpret_cast<W*>(_w)->bitcount; };

  auto entry = (u4)(C::ChunkSize * blockIdx.x);
  auto allowed_len = min((u4)C::ChunkSize, inlen - entry);

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

  if (p_bits <= C::BITWIDTH) {
    s_reduced[threadIdx.x] = p_reduced;
    s_bitcount[threadIdx.x] = p_bits;
  }
  else {
    // Spill all ShardSize originals to sparse buffer; encode shard as alt_code.
    s_reduced[threadIdx.x] = alt_code;
    s_bitcount[threadIdx.x] = alt_bitcount;

    auto start_idx = atomicAdd(sp_count, (u4)C::ShardSize);
    for (auto i = 0; i < C::ShardSize; i++) {
      auto idx = (threadIdx.x * C::ShardSize) + i;
      auto gid = entry + idx;
      if (gid < inlen) {
        sp_val[start_idx + i] = (typename C::T)s_to_encode[idx];
        sp_idx[start_idx + i] = gid;
      }
    }
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------
// Phase 3: shuffle-merge
//
// Iteratively merges adjacent shard pairs.  Each round halves the number of
// live shards by bit-packing the right subgroup into the left.
// Follows EIP's v1_shuffle_merge_sync__constexpr_shardsize pattern.
// ---------------------------------------------------------------------------
template <class C>
__forceinline__ __device__ void hfr_shuffle_merge(
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

    // Clear the right-subgroup slot before or-ing into the destination region.
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

// ---------------------------------------------------------------------------
// Phase 4: sequential data store
//
// Block i writes its packed words starting at dn_out[C::ChunkSize * i].
// No atomicAdd / random-access needed: the decoder can locate each block's
// bitstream from dn_bitcount[blockIdx.x] and the fixed stride C::ChunkSize.
// ---------------------------------------------------------------------------
template <class C>
__forceinline__ __device__ void hfr_data_store(
    volatile typename C::Hf* s_reduced, volatile u4* s_bitcount, typename C::Hf* dn_out,
    u4* dn_bitcount)
{
  const auto bc_this_block = s_bitcount[0];
  const auto n_cell = (bc_this_block + C::BITWIDTH - 1) / C::BITWIDTH;

  if (threadIdx.x < n_cell)
    dn_out[C::ChunkSize * blockIdx.x + threadIdx.x] = (typename C::Hf)s_reduced[threadIdx.x];

  if (threadIdx.x == 0) dn_bitcount[blockIdx.x] = bc_this_block;
}

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------
template <class C, int return_at = HFReVISIT_disable_trap>
__global__ void KERNEL_CUHIP_HFReVISIT_encode(
    typename C::T* in, u4 inlen, typename C::Hf* dram_book, u4 bklen, typename C::Hf alt_code,
    u4 alt_bitcount, typename C::Hf* dn_out, u4* dn_bitcount, typename C::T* sp_val, u4* sp_idx,
    u4* sp_count)
{
  static_assert(C::ReduceTimes >= 1, "ReduceTimes must be >= 1.");
  static_assert((2 << C::Magnitude) < 98304, "Shared memory exceeds limit.");

  __shared__ typename C::T s_to_encode[C::ChunkSize];
  __shared__ typename C::Hf s_book[1024];
  __shared__ typename C::Hf s_reduced[C::NumShards];
  __shared__ u4 s_bitcount[C::NumShards + 1];

  hfr_data_load<C>(in, inlen, dram_book, bklen, s_to_encode, s_book);
  RETURN_AT(1);

  hfr_reduce_merge_compact<C>(
      inlen, alt_code, alt_bitcount, s_to_encode, s_book, s_reduced, s_bitcount, sp_val, sp_idx,
      sp_count);
  RETURN_AT(2);

  hfr_shuffle_merge<C>(s_reduced, s_bitcount);
  RETURN_AT(3);

  hfr_data_store<C>(s_reduced, s_bitcount, dn_out, dn_bitcount);
}

}  // namespace phf

// ---------------------------------------------------------------------------
// Module-level GPU dispatch
// ---------------------------------------------------------------------------
namespace phf::module {

template <typename T, int Magnitude, int ReduceTimes, bool UseScan, typename Hf>
int HFReVISIT_encode<T, Magnitude, ReduceTimes, UseScan, Hf>::GPU_kernel(
    T* in, size_t len, Hf* bk, u4 bklen, Hf alt_code, u4 alt_bitcount, Hf* dn_out, u4* dn_bitcount,
    T* sp_val, u4* sp_idx, u4* sp_count, void* stream)
{
  using C = HFReVISIT_config<T, Magnitude, ReduceTimes, Hf>;

  constexpr auto nthread = C::BlockDim;
  const auto nblock = (len - 1) / C::ChunkSize + 1;

  phf::KERNEL_CUHIP_HFReVISIT_encode<C><<<nblock, nthread, 0, (cudaStream_t)stream>>>(
      in, (u4)len, bk, bklen, alt_code, alt_bitcount, dn_out, dn_bitcount, sp_val, sp_idx,
      sp_count);

  return 0;
}

}  // namespace phf::module

// ---------------------------------------------------------------------------
// Explicit instantiations (split across rs_merge_rN.cu for parallel builds)
// ---------------------------------------------------------------------------

#define __INSTANTIATE_HFR(T, MAG, RED, SCAN) \
  template struct phf::module::HFReVISIT_encode<T, MAG, RED, SCAN, u4>;

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
