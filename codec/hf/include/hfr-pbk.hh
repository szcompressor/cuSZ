#ifndef PSZ_HFR_PBK_HH
#define PSZ_HFR_PBK_HH

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "c_type.h"
#include "cusz/component.hh"
#include "hf.h"
#include "mem/cxx_backends.h"

using Hf = u4;

enum class IncompPredKind : u1 { None = 0, Lorenzo1D, Lorenzo2D, Lorenzo3D };

struct IncompRedo {
  void const* in_data = nullptr;
  float ebx2_r = 0;
  u2 radius = 0;
  u4 leapy = 0, leapz = 0;
  u4 dimy = 0, dimz = 0;  // extent.y/.z, for the tile-order boundary guard (partial tiles)
  IncompPredKind kind = IncompPredKind::None;
  // eq is tile-ordered (2D/3D HFR-family): map the in-tile offset back to the
  // linear gid before recomputing. linear eq (1D, HFr2) leaves this false.
  bool nd_tile = false;
};

namespace psz {

constexpr u4 log2_floor(u4 n);
constexpr u4 log2_ceil(u4 n);

template <size_t _Magnitude>
struct _parameterized_hfr_pbk_constants;

struct HFR_PBK_Constants;

template <u2 _Radius, u1 _NumBooks>
struct HFR_PBK_Config;

template <int Seq>
struct HFR_PBK_Launch;

}  // namespace psz

constexpr u4 psz::log2_floor(u4 n)
{
  return n == 0 ? throw "n must be > 0" : (n < 2) ? 0 : 1 + log2_floor(n >> 1);
}

constexpr u4 psz::log2_ceil(u4 n)
{
  return n == 0 ? throw "n must be > 0" : (n & (n - 1)) == 0 ? log2_floor(n) : 1 + log2_floor(n);
}

// The current header can support up to 13 bits.
// Need to reconcile with other-configured data chunksize.
template <size_t _Magnitude>
struct psz::_parameterized_hfr_pbk_constants {
  static constexpr u1 NumBooks = 25;
  static constexpr u1 Radius = 128;

  static constexpr size_t Magnitude = _Magnitude;
  static constexpr size_t BlockSize = 1u << Magnitude;
  static constexpr size_t NumCoding = 32;
  static constexpr size_t MaxRadius = 128;
  static constexpr size_t MaxDictsize = MaxRadius * 2;
  // prebuilt PBK25_R128 reverse-book: first[32]·u4 + entry[32]·u4 + keys[256]·u1
  static constexpr size_t RvbkBytesPerBook = 512;

  // Default ReduceTimes (r3 preset); RT-parameterized variants below.
  static constexpr size_t ReduceTimes = 3;

  static constexpr u4 MASK_BREAKS = 0x00FF0000u;
  static constexpr u4 MASK_UNPRED = 0x0000FF00u;
  static constexpr u4 MASK_TF = 0x000000FFu;

  // header
  static constexpr size_t BitsMaxNumUnpred = 3;
  static constexpr size_t BitsMaxNumBreaks = 6;
  static constexpr size_t BitsEncId = 5;
  static constexpr size_t BitsDense =
      32 - (BitsEncId + BitsMaxNumUnpred + BitsMaxNumBreaks);  // = 18

  static constexpr size_t MaxNumUnpred = (1 << BitsMaxNumUnpred) - 1;  // >=7 -> incomp.unpred
  static constexpr size_t MaxNumBreaks = (1 << BitsMaxNumBreaks) - 1;  // >=63 -> incomp.breaks

  // up to MaxNumUnpred cells (compact_cell<f4,u2>=6B), word-padded.
  static constexpr size_t OutlierCellBytes = sizeof(_ptb::compact_cell<f4, u2>);  // 6B
  static constexpr size_t MaxUnpredBytes = (MaxNumUnpred * OutlierCellBytes + 3) & ~size_t(3);
  static constexpr size_t MaxUnpredWords = MaxUnpredBytes / sizeof(u4);

  // Worst-case per-block: (breaks, Hf=u4, BreakCell=4B) + outliers
  static constexpr size_t StridePerBlockWords = BlockSize + MaxNumBreaks + MaxUnpredWords;
  static constexpr size_t StridePerBlockBytes = StridePerBlockWords * sizeof(uint32_t);
  static constexpr size_t CodeIncompUnpred = 31;
  static constexpr size_t CodeIncompBreaks = 30;

  // worst-case bitstream
  static_assert(BlockSize * 16 < (1ull << BitsDense), "Magnitude exceeds bits(18) budget");
  static_assert(Magnitude <= 16, "block-local idx must fit uint16_t");
};

// preset-10 in use (deterministic perf); re-profile if changed.
struct psz::HFR_PBK_Constants : psz::_parameterized_hfr_pbk_constants<10> {};

// Include after HFR_PBK_Constants: the companion's HFR_Opts reads ReduceTimes.
#include "hfr-pbk_ver.hh"

namespace psz {

// n_unpred 6B-compact_cell<f4, u2>, 4B-aligned
__host__ __device__ __forceinline__ u4 pbk_unpred_bytes(u4 n_unpred)
{
  return (n_unpred * (u4)sizeof(_ptb::compact_cell<f4, u2>) + 3u) & ~3u;
}

__host__ __device__ __forceinline__ u4 pbk_unpred_words(u4 n_unpred)
{
  return pbk_unpred_bytes(n_unpred) / 4u;  // H4 words
}

// RT-parameterized constants (adds shard-fan-out fields).
template <int _RT>
struct HFR_PBK_Constants_RT : public HFR_PBK_Constants {
  static constexpr size_t ReduceTimes = (size_t)_RT;
  static constexpr size_t ShardSize = 1u << ReduceTimes;           // points per thread
  static constexpr size_t NumShards = BlockSize / ShardSize;       // threads per block
  static constexpr size_t ShuffleTimes = Magnitude - ReduceTimes;  // log2(NumShards)
  static_assert(ReduceTimes <= Magnitude, "ReduceTimes must be <= Magnitude.");
};

using HFR_PBK_Constants_r2 = HFR_PBK_Constants_RT<2>;
using HFR_PBK_Constants_r3 = HFR_PBK_Constants_RT<3>;
using HFR_PBK_Constants_r4 = HFR_PBK_Constants_RT<4>;

}  // namespace psz

template <u2 _Radius, u1 _NumBooks>
struct psz::HFR_PBK_Config {
  static_assert(_NumBooks <= 25, "NumBooks must be <= 25.");
  static constexpr u2 Radius = _Radius;
  static constexpr u2 Bklen = _Radius * 2;
};

template <int Seq>
struct psz::HFR_PBK_Launch {
  static constexpr dim3 tile = dim3(1024, 1, 1);
  static constexpr dim3 sequentiality = dim3(Seq, 1, 1);  // x-sequentiality == 4
  static constexpr dim3 seq = sequentiality;
  static constexpr dim3 thread_block = dim3(1024 / Seq, 1, 1);
  static dim3 thread_grid(dim3 len3)
  {
    auto div3 = [](dim3 l, dim3 subl) {
      return dim3((l.x - 1) / subl.x + 1, (l.y - 1) / subl.y + 1, (l.z - 1) / subl.z + 1);
    };

    return div3(len3, tile);
  };

  using Perf = psz::PredictorTile<tile.x, seq.x, tile.y, seq.y, tile.z, seq.z>;
};

namespace psz {

struct _future_unpred_t {
  uint16_t val_raw;
  uint16_t idx : psz::log2_ceil(
      psz::HFR_PBK_Constants::BlockSize);  // ideally only 10 bits to track the block
} __attribute__((packed));

template <u2 Radius>
struct HFR_PBK_Breaks {
  static_assert(Radius <= psz::HFR_PBK_Constants::MaxRadius, "Radius must be <= 128.");
  uint16_t val;
  uint16_t idx;
} __attribute__((packed));

}  // namespace psz

namespace psz::_future {

template <typename T, u2 Radius>
struct bheader {
  using C = HFR_PBK_Constants;

  union {
    u4 _tuple4;
    struct {
      u4 n_unpred : C::BitsMaxNumUnpred;
      u4 n_breaks : C::BitsMaxNumBreaks;
      u4 enc_id : C::BitsEncId;
      u4 bits : 32 - (C::BitsEncId + C::BitsMaxNumUnpred + C::BitsMaxNumBreaks);
    };
  };
  u4 entry : 32;  // larger scale

} __attribute__((packed));

}  // namespace psz::_future

#endif  // PSZ_HFR_PBK_HH
