#include <cassert>
#include <iostream>

#include "hf_impl.hh"
#include "mem/cxx_sp_cpu.h"
#include "mem/cxx_sp_gpu.h"
#include "rs_merge.hh"

using namespace std;

using u4 = uint32_t;
using u2 = uint16_t;

#define PARFOR1_BLOCK_RSMERGE() for (auto thread_idx = 0; thread_idx < C::BlockDim; thread_idx++)

// enumerate states for all "threads"
#define SHFLMERGE_THREAD_STATE_ENUMERATION                                   \
  std::unique_ptr<u4[]> _l = std::make_unique<u4[]>(C::NumShards);           \
  std::unique_ptr<u4[]> _r = std::make_unique<u4[]>(C::NumShards);           \
  std::unique_ptr<u4[]> _lbc = std::make_unique<u4[]>(C::NumShards);         \
  std::unique_ptr<u4[]> _used__units = std::make_unique<u4[]>(C::NumShards); \
  std::unique_ptr<u4[]> _used___bits = std::make_unique<u4[]>(C::NumShards); \
  std::unique_ptr<u4[]> _unused_bits = std::make_unique<u4[]>(C::NumShards); \
  std::unique_ptr<u4[]> _this_point = std::make_unique<u4[]>(C::NumShards);  \
  std::unique_ptr<u4[]> _lsym = std::make_unique<u4[]>(C::NumShards);        \
  std::unique_ptr<u4[]> _rsym = std::make_unique<u4[]>(C::NumShards);        \
  std::unique_ptr<Hf*[]> _lend = std::make_unique<Hf*[]>(C::NumShards);

// mimick GPU programming
#define l _l[thread_idx]
#define r _r[thread_idx]
#define lbc _lbc[thread_idx]
#define used__units _used__units[thread_idx]
#define used___bits _used___bits[thread_idx]
#define unused_bits _unused_bits[thread_idx]
#define this_point _this_point[thread_idx]
#define lsym _lsym[thread_idx]
#define rsym _rsym[thread_idx]
#define lend _lend[thread_idx]

namespace phf {

template <typename C>
void KERNEL_SEQ_HFReVISIT_encode(
    typename C::T* in, u4 inlen,               // input data and length
    typename C::Hf* dram_book, u4 bklen,       // Huffman codebook and its length
    typename C::Hf alt_code, u4 alt_bitcount,  // alternative code for breaking points
    typename C::Hf* dn_out, u4* dn_bitcount,   // dense output and bit counts
    u4* dn_start_loc, u4* loc_inc,             // dense start locations
    void* sp_val_idx, u4* sp_count             // sparse outputs
)
{
  using T = typename C::T;
  using Hf = typename C::Hf;
  using W = typename C::W;
  using Cell = typename C::Cell;

  auto bits_of = [](Hf* _w) { return reinterpret_cast<W*>(_w)->bitcount; };

  auto num_blocks = (inlen - 1) / C::ChunkSize + 1;

  for (auto block_idx = 0; block_idx < num_blocks; block_idx++) {
    auto entry = [&]() { return C::ChunkSize * block_idx; };
    auto allowed_len = [&]() { return min(C::ChunkSize, inlen - entry()); };
    auto neutral_val = [&]() { return bklen / 2; };

    typename C::T to_encode[C::ChunkSize] = {};
    typename C::Hf reduced[C::NumShards] = {};
    u4 bitcount[C::NumShards + 1] = {};
    u4 s_start_loc = 0;

    // Lambda: data loading
    auto CPU_Lambda__data_load = [&]() {
      for (auto i = 0; i < C::ChunkSize; i++)
        to_encode[i] = i < allowed_len() ? in[entry() + i] : neutral_val();
    };

    // Lambda: reduce merge with compaction
    auto CPU_Lambda__reduce_merge_compact = [&]() {
      // C::NumShards = C::BlockDim
      PARFOR1_BLOCK_RSMERGE()
      {
        auto p_bits{0u};
        Hf p_reduced{0x0};

        // per-thread loop, merge
        for (auto i = 0; i < C::ShardSize; i++) {
          auto idx = (thread_idx * C::ShardSize) + i;

          auto p_key = to_encode[idx];
          auto p_val = dram_book[p_key];
          auto sym_bits = bits_of(&p_val);

          p_val <<= (C::BITWIDTH - sym_bits);
          p_reduced |= (p_val >> p_bits);
          /* p_bits += sym_bits; (semantically so, subs is below) */
          p_bits += sym_bits * (idx < allowed_len());
        }

        // After a thread is done, check the breaking points, record the merged bits & bit count.
        if (p_bits <= C::BITWIDTH) {
          reduced[thread_idx] = p_reduced;
          bitcount[thread_idx] = p_bits;
        }
        else {  // write out breaking points and replace them
          reduced[thread_idx] = alt_code;
          bitcount[thread_idx] = alt_bitcount;

          auto start_idx = (*sp_count);
          for (auto i = 0; i < C::ShardSize; ++i) {
            auto idx = (thread_idx * C::ShardSize) + i;
            auto gid = C::ChunkSize * block_idx + idx;
            if (gid < inlen) { (Cell*)sp_val_idx[start_idx + i] = {to_encode[idx], gid}; }
          }
          *sp_count += C::ShardSize;
        }
        //
        // cout << thread_idx << "\t" << bitset<32>(reduced[thread_idx])
        //      << "\tbc=" << bitcount[thread_idx] << endl;
      }
      //
    };

    // Lambda: shuffle merge
    auto CPU_Lambda__shuffle_merge = [&]() {
      for (auto sf = C::ShuffleTimes, stride = 1u; sf > 0; sf--, stride *= 2) {
        //
        SHFLMERGE_THREAD_STATE_ENUMERATION;
        PARFOR1_BLOCK_RSMERGE()
        {
          l = thread_idx / (stride * 2) * (stride * 2);
          r = l + stride;

          // left chunk details
          lbc = bitcount[l];  // left bit count
          used__units = lbc / C::BITWIDTH;
          used___bits = lbc % C::BITWIDTH;
          unused_bits = C::BITWIDTH - used___bits;

          // destination
          // Hf* lend = (Hf*)(reduced + l + used__units);
          lend = (Hf*)(reduced + l + used__units);

          // use lsym and rsym to deteremine what to merge
          this_point = reduced[thread_idx];
          lsym = this_point >> used___bits;
          rsym = this_point << unused_bits;

          // if (thread_idx >= r and thread_idx < r + stride)
          //   std::cout << "CPU Thread idx: " << thread_idx << ", l: " << l << ", r: " << r
          //             << ", lsym: " << lsym << ", rsym: " << rsym << std::endl;
        }

        PARFOR1_BLOCK_RSMERGE()
        {
          // clear up the out-of-bound for merge
          // such that tailing threads or-op empty bits (lazy)
          if (thread_idx >= r and thread_idx < r + stride) { reduced[thread_idx] = 0; }
        }

        PARFOR1_BLOCK_RSMERGE()
        {
          // use the whole right subgroup to update bits in the destination
          if (thread_idx >= r and thread_idx < r + stride) {
            lend[0] |= lsym;  // cross data unit: left unit
            lend[1] |= rsym;  // cross data unit: right unit

            // cout << "CPU Thread idx: " << thread_idx << ", lend[0]: " << bitset<32>(lend[0])
            //      << ", lend[1]: " << bitset<32>(lend[1]) << std::endl;
          }

          // update metadata
          if (thread_idx == l) bitcount[l] += bitcount[r];
        }
      }
    };

    auto CPU_Lambda__data_store = [&]() {
      constexpr auto write_output_ver = 2;
      const auto bc_this_block = bitcount[0];
      auto n_cell = (bc_this_block - 1) / C::BITWIDTH + 1;

      if constexpr (write_output_ver == 1) {
        PARFOR1_BLOCK_RSMERGE()
        {
          if (thread_idx < n_cell) {
            auto dram_addr = C::ChunkSize * block_idx + thread_idx;
            dn_out[dram_addr] = reduced[thread_idx];
          }
        }
        dn_bitcount[block_idx] = bc_this_block;
      }
      else if constexpr (write_output_ver == 2) {
        // Atomic ops are not "that" meaningful when not parallelized.
        const auto start_loc = *loc_inc;
        *loc_inc += n_cell;

        PARFOR1_BLOCK_RSMERGE()
        {
          // ceil(bitcount / bits_of(Hf)) cannot be greater than blockDim.x
          if (thread_idx < n_cell) dn_out[start_loc + thread_idx] = reduced[thread_idx];
        }

        // TODO change dn_bitcount to uint16_t*
        // decomp loc is known (fixed) according to blockIdx.x
        dn_start_loc[block_idx] = start_loc;
        dn_bitcount[block_idx] = bc_this_block;
      }
      else {
        printf("write_output_ver must be 1 or 2");
      }
    };

    // -----------------------
    CPU_Lambda__data_load();             // done
    CPU_Lambda__reduce_merge_compact();  // done
    CPU_Lambda__shuffle_merge();         // done
    CPU_Lambda__data_store();            // TO VERIFY
    // -----------------------
  }
}

}  // namespace phf

namespace phf::module {

// TODO add dn_bitcount length
// template <typename T, int Magnitude, int ReduceTimes, bool use_scan, typename Hf>
template <class T, class C>
void CPU_HFReVISIT_encode(
    T* in, const size_t len, phf::book<typename C::Hf> book, phf::dense<typename C::Hf> dn,
    void* _sp)
{
  // using C = HFReVISIT_config<T, Magnitude, ReduceTimes, Hf>;

  constexpr auto nthread = 1 << C::ShuffleTimes;
  auto slab_size = C::ChunkSize;
  auto nblock = (len - 1) / slab_size + 1;

  using Compact = _portable::compact_CPU<T>;
  auto sp = (Compact*)_sp;

  if (false) {
    std::cout << "nthread: " << nthread << std::endl;
    std::cout << "slab_size: " << slab_size << std::endl;
    std::cout << "nblock: " << nblock << std::endl;

    // printf("in: %p\n", static_cast<const void*>(in.buf));
    printf("bk: %p\n", static_cast<const void*>(book.bk));
    printf("dn.out: %p\n", static_cast<void*>(dn.encoded));
    printf("dn.bitcount: %p\n", static_cast<void*>(dn.chunk_nbit));
    printf("dn.start_loc: %p\n", static_cast<void*>(dn.chunk_loc));
    // printf("sp.val: %p\n", static_cast<void*>(sp->val()));
    // printf("sp.idx: %p\n", static_cast<void*>(sp->idx()));
    // printf("sp.num: %p\n", static_cast<void*>(sp->num()));
  }

  phf::KERNEL_SEQ_HFReVISIT_encode<C>(
      in, len, book.bk, book.bklen, book.alt_prefix_code, book.alt_bitcount, dn.encoded,
      dn.chunk_nbit, dn.chunk_loc, dn.loc_inc, sp.val, sp.idx, sp.num);
}

}  // namespace phf::module

#undef SHFLMERGE_THREAD_STATE_ENUMERATION
#undef l
#undef r
#undef lbc
#undef used__units
#undef used___bits
#undef unused_bits
#undef this_point
#undef lsym
#undef rsym

// #define __INSTANTIATE_RSMERGE_4(T, MAG, RED, SCAN)                    \
//   template void phf::module::CPU_HFReVISIT_encode<T, MAG, RED, SCAN>( \
//       phf::array<T> in, phf::book<u4> book, phf::dense<u4> dn, phf::sparse<T> sp);

// // TODO disable u4
// #define __INSTANTIATE_RSMERGE_3(MAG, RED, SCAN) \
//   __INSTANTIATE_RSMERGE_4(u4, MAG, RED, SCAN)   \
//   __INSTANTIATE_RSMERGE_4(u2, MAG, RED, SCAN)   \
//   __INSTANTIATE_RSMERGE_4(u1, MAG, RED, SCAN)

// // TODO choose the best-fit
// #define __INSTANTIATE_RSMERGE_2(RED, SCAN) \
//   __INSTANTIATE_RSMERGE_3(12, RED, SCAN)   \
//   __INSTANTIATE_RSMERGE_3(11, RED, SCAN)   \
//   __INSTANTIATE_RSMERGE_3(10, RED, SCAN)   \
//   __INSTANTIATE_RSMERGE_3(9, RED, SCAN)    \
//   __INSTANTIATE_RSMERGE_3(8, RED, SCAN)    \
//   __INSTANTIATE_RSMERGE_3(7, RED, SCAN)    \
//   __INSTANTIATE_RSMERGE_3(6, RED, SCAN)    \
//   __INSTANTIATE_RSMERGE_3(5, RED, SCAN)

// #define __INSTANTIATE_RSMERGE_1(RED)  \
//   __INSTANTIATE_RSMERGE_2(RED, false) \
//   __INSTANTIATE_RSMERGE_2(RED, true)
