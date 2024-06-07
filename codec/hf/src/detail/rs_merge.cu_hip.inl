/**
 * @file rmsm.cu
 * @author Jiannan Tian
 * @brief Instantiation of RMSM
 * @version 0.1
 * @date 2020-05-30
 *
 * (C) 2020 by Washington State University, Argonne National Laboratory
 *
 */

#include <cstdio>
#include <iostream>

#include "auxiliary.inl"
#include "hf_type.h"
#include "rs_merge.hxx"

using std::cout;
using std::endl;
using u4 = uint32_t;

#define DEBUG_1_DATA_LOADING(...)                   \
  if constexpr (debug == 1) {                       \
    if (blockIdx.x == 0 and threadIdx.x == 0)       \
      printf("HFR kernel ends at data-loading.\n"); \
    return;                                         \
  }

#define DEBUG_2_REDUCE_MERGE_COMPACT(...)                   \
  if constexpr (debug == 2) {                               \
    if (blockIdx.x == 0 and threadIdx.x == 0)               \
      printf("HFR kernel ends at reduce-merge-compact.\n"); \
    return;                                                 \
  }

#define DEBUG_3_SHUFFLE_MERGE_WRITE(...)                   \
  if constexpr (debug == 3) {                              \
    if (blockIdx.x == 0 and threadIdx.x == 0)              \
      printf("HFR kernel ends at shuffle-merge-write.\n"); \
    return;                                                \
  }

#define RETURN_AT(BreakPoint) \
  if constexpr (return_at == BreakPoint) return;

namespace phf {

namespace {

// interrupt to debug
static const int HFReVISIT_disable_trap = 0;
static const int HFReVISIT_trap_dataloading = 1;
static const int HFReVISIT_trap_reduce = 2;
static const int HFReVISIT_trap_shuffle = 3;
}  // namespace

template <
    class C, int return_at = HFReVISIT_disable_trap,
    int debug = HFReVISIT_disable_trap>
__global__ void KERNEL_CUHIP_HFReVISIT_encode(
    INPUT typename C::T* in, u4 inlen, typename C::Hf* dram_book, u4 bklen,  //
    INPUT typename C::Hf alt_code, u4 alt_bitcount,                          //
    OUTPUT typename C::Hf* dn_out, u2* dn_bitcount, u4* dn_start_loc,
    u4* loc_inc,                                             //
    OUTPUT typename C::T* sp_val, u4* sp_idx, u4* sp_count,  //
    DEBUG u4 debug_blockid = 0)
{
  using T = typename C::T;
  using Hf = typename C::Hf;
  using W = typename C::W;

  static_assert(
      C::ReduceTimes >= 1,
      "Reduction Factor must be >= 1, otherwise, you lose the point of "
      "compression.");
  static_assert((2 << C::Magnitude) < 98304, "Shared memory used too much.");

  // __shared__ typename C::ChunkType chunk;
  __shared__ T to_encode[C::ChunkSize];
  __shared__ Hf reduced[C::NumShards];
  __shared__ Hf book[1024];
  __shared__ u4 bitcount[C::NumShards + 1];
  // __shared__ u4 locs[C::NumShards + 1];  // TODO merge bits and locs
  __shared__ u4 s_start_loc;

  auto total_threads = [&]() { return blockDim.x; };
  auto bits_of = [](Hf* _w) { return reinterpret_cast<W*>(_w)->bitcount; };
  auto entry = [&]() { return C::ChunkSize * blockIdx.x; };
  auto allowed_len = [&]() { return min(C::ChunkSize, inlen - entry()); };

  auto data_loading = [&]() {
    // one-time setup: load book
    for (auto i = threadIdx.x; i < bklen; i += blockDim.x)
      book[i] = dram_book[i];
    // __syncthreads();

    // initialize padded values (last-block)
    // avoid explicit memset of __shared__ array
    // otherwise, the random value from the unintialized __shared__ array
    // (to_encode) corrupts the book lookup
    for (auto i = threadIdx.x; i < C::ChunkSize; i += blockDim.x)
      to_encode[i] = i < allowed_len() ? in[entry() + i] : bklen / 2;
    __syncthreads();
  };

  auto reduce_merge_compact = [&]() {
    auto p_bits{0u};
    Hf p_reduced{0x0};

    // bit layout: accumulate from MSB to LSB
    for (auto i = 0; i < C::ShardSize; i++) {
      auto _local_id = [&]() { return i + threadIdx.x * C::ShardSize; };

      auto p_key = to_encode[_local_id()];
      auto p_val = book[p_key];
      auto sym_bits = bits_of(&p_val);

      p_val <<= C::BITWIDTH - sym_bits;
      p_reduced |= (p_val >> p_bits);
      // p_bits += sym_bits;
      p_bits += sym_bits * (_local_id() < allowed_len());
    }

    if (p_bits <= C::BITWIDTH) {  // no breaking points
      reduced[threadIdx.x] = p_reduced;
      bitcount[threadIdx.x] = p_bits;
    }
    else {  // write out breaking points and replace them
      reduced[threadIdx.x] = alt_code;
      bitcount[threadIdx.x] = alt_bitcount;

      auto start_idx = atomicAdd(sp_count, C::ShardSize);
      for (auto i = 0; i < C::ShardSize; i++) {
        auto local_id = i + threadIdx.x * C::ShardSize;
        auto gid = C::ChunkSize * blockIdx.x + local_id;

        if (gid < inlen) {
          sp_val[start_idx + i] = to_encode[local_id];
          sp_idx[start_idx + i] = gid;
        }
      }
    }
    __syncthreads();
  };

  auto shuffle_merge_and_write = [&]() {
    auto stride = 1;

    for (auto sf = C::ShuffleTimes; sf > 0; sf--, stride *= 2) {
      auto l = threadIdx.x / (stride * 2) * (stride * 2);
      auto r = l + stride;
      auto lbc = bitcount[l];
      u4 dtype_ofst = lbc / C::BITWIDTH;
      u4 used_bits = lbc % C::BITWIDTH;
      u4 unused_bits = C::BITWIDTH - used_bits;
      auto lend = (Hf*)(reduced + l + dtype_ofst);

      auto this_point = reduced[threadIdx.x];
      auto lsym = this_point >> used_bits;
      auto rsym = this_point << unused_bits;

      // threadIdx.x in [r, r+stride) or ((r ..< r+stride )), whole right
      // subgroup &reduced[ ((r ..< r+stride)) ] have conflicts with (lend+
      // ((0 ..< stride)) + 0/1) because the whole shuffle subprocedure
      // compresses at a factor of < 2
      if (threadIdx.x >= r and threadIdx.x < r + stride)
        atomicAnd((Hf*)(reduced + threadIdx.x), 0x0);
      __syncthreads();

      // whole right subgroup
      if (threadIdx.x >= r and threadIdx.x < r + stride) {
        atomicOr(
            lend + (threadIdx.x - r) + 0,
            lsym);  // <=> *(lend + (threadIdx.x -r) + 0) = lsym;
        atomicOr(
            lend + (threadIdx.x - r) + 1,
            rsym);  // <=> *(lend + (threadIdx.x -r) + 0) = lsym;
      }
      ///* optional */ __syncthreads();

      if (threadIdx.x == l)
        bitcount[l] += bitcount[l + stride];  // very imbalanced
      __syncthreads();
    }

    constexpr auto write_output_ver = 2;
    if constexpr (write_output_ver == 1) {
      auto bitcount_this_block = bitcount[0];
      auto n_cell = (bitcount_this_block - 1) / C::BITWIDTH + 1;
      if (threadIdx.x < n_cell) {
        auto dram_addr = C::ChunkSize * blockIdx.x + threadIdx.x;
        dn_out[dram_addr] = reduced[threadIdx.x];
      }
      if (threadIdx.x == 0) dn_bitcount[blockIdx.x] = bitcount_this_block;
    }
    else if constexpr (write_output_ver == 2) {
      auto bitcount_this_block = bitcount[0];
      auto n_cell = (bitcount_this_block - 1) / C::BITWIDTH + 1;

      if (threadIdx.x == 0) auto s_start_loc = atomicAdd(loc_inc, n_cell);
      __syncthreads();

      auto start_loc = s_start_loc;

      // ceil(bitcount / bits_of(Hf)) cannot be greater than blockDim.x
      if (threadIdx.x < n_cell) dn_out[start_loc] = reduced[threadIdx.x];

      // TODO change dn_bitcount to uint16_t*
      if (threadIdx.x == 0) {
        // decomp loc is known (fixed) according to blockIdx.x
        dn_start_loc[blockIdx.x] = start_loc;
        dn_bitcount[blockIdx.x] = bitcount_this_block;
      }
    }
    else {
      if (blockIdx.x == 0 and threadIdx.x == 0)
        printf("write_output_ver must be 1 or 2");
    }
  };

  ////////////////////////////////////////

  data_loading();

  DEBUG_1_DATA_LOADING();
  RETURN_AT(1);

  reduce_merge_compact();

  DEBUG_2_REDUCE_MERGE_COMPACT();
  RETURN_AT(2);

  shuffle_merge_and_write();

  DEBUG_3_SHUFFLE_MERGE_WRITE();
  RETURN_AT(3);
}

}  // namespace phf

////////////////////////////////////////////////////////////////////////////////

namespace phf::cuhip {

// TODO add dn_bitcount length
template <
    typename T, int Magnitude, int ReduceTimes, bool use_scan, typename Hf>
void GPU_HFReVISIT_encode(
    INPUT hfcxx_array<T> in, hfcxx_book<Hf> book,    //
    OUTPUT hfcxx_dense<Hf> dn, hfcxx_compact<T> sp,  //
    GPU_QUEUE void* stream, DEBUG u4 debug_blockid)
{
  using C = HFReVISIT_config<T, Magnitude, ReduceTimes, Hf>;

  constexpr auto nthread = 1 << C::ShuffleTimes;
  auto slab_size = C::ChunkSize;
  auto nblock = (in.len - 1) / slab_size + 1;

  if (true) {
    cout << "nthread: " << nthread << endl;
    cout << "slab_size: " << slab_size << endl;
    cout << "nblock: " << nblock << endl;

    printf("in: %p\n", in.buf);
    printf("bk: %p\n", book.bk.buf);
    printf("dn.out: %p\n", dn.out);
    printf("dn.bitcount: %p\n", dn.bitcount);
    printf("dn.start_loc: %p\n", dn.start_loc);
    printf("sp.val: %p\n", sp.val);
    printf("sp.idx: %p\n", sp.idx);
    printf("sp.num: %p\n", sp.num);
  }

  phf::KERNEL_CUHIP_HFReVISIT_encode<C>
      <<<nblock, nthread, 0, (cudaStream_t)stream>>>(
          in.buf, in.len, book.bk.buf, book.bk.len, book.alt_prefix_code,
          book.alt_bitcount, dn.out, dn.bitcount, dn.start_loc, dn.loc_inc,
          sp.val, sp.idx, sp.num);

  cudaStreamSynchronize((cudaStream_t)stream);
}

}  // namespace phf::cuhip

#define __INSTANTIATE_RSMERGE_4(T, MAG, RED, SCAN)                            \
  template void phf::cuhip::GPU_HFReVISIT_encode<T, MAG, RED, SCAN>(         \
      INPUT hfcxx_array<T> in, hfcxx_book<u4> book,                           \
      OUTPUT hfcxx_dense<u4> dn, hfcxx_compact<T> sp, GPU_QUEUE void* stream, \
      DEBUG u4 debug_blockid);

// TODO disable u4
#define __INSTANTIATE_RSMERGE_3(MAG, RED, SCAN) \
  __INSTANTIATE_RSMERGE_4(u4, MAG, RED, SCAN)   \
  __INSTANTIATE_RSMERGE_4(u2, MAG, RED, SCAN)   \
  __INSTANTIATE_RSMERGE_4(u1, MAG, RED, SCAN)

// TODO choose the best-fit
#define __INSTANTIATE_RSMERGE_2(RED, SCAN) \
  __INSTANTIATE_RSMERGE_3(12, RED, SCAN)   \
  __INSTANTIATE_RSMERGE_3(11, RED, SCAN)   \
  __INSTANTIATE_RSMERGE_3(10, RED, SCAN)

#define __INSTANTIATE_RSMERGE_1(RED)  \
  __INSTANTIATE_RSMERGE_2(RED, false) \
  __INSTANTIATE_RSMERGE_2(RED, true)
