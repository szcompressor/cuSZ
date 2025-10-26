#ifndef AF64552A_247F_47E8_BFFA_EFC88E0491EF
#define AF64552A_247F_47E8_BFFA_EFC88E0491EF

#include <stdexcept>

#include "hf_impl.hh"
#include "mem/sp_interface.h"

#define SUBROUNTINE __device__ __forceinline__
// #define SUBROUNTINE __device__
#define KERNEL __global__ void
#define SHARED volatile
#define THREAD_PRIVATE
#define INPUT
#define OUTPUT
#define DEBUG
#define GPU_STREAM
#define GPU_QUEUE
#define PERF_TUNE

using u4 = uint32_t;

template <typename _T, int _Magnitude, int _ReduceTimes, typename _Hf = u4>
struct HFReVISIT_config {
  using T = _T;
  using Hf = _Hf;
  using W = HuffmanWord<sizeof(Hf)>;
  using Cell = _portable::compact_cell<_T>;

  static constexpr bool DBG = false;
  static constexpr u4 Magnitude = _Magnitude;
  static constexpr u4 ReduceTimes = _ReduceTimes;
  static constexpr u4 ShuffleTimes = Magnitude - ReduceTimes;
  static constexpr u4 ChunkSize = 1 << Magnitude;
  static constexpr u4 ShardSize = 1 << ReduceTimes;
  static constexpr u4 NumShards = 1 << ShuffleTimes;
  static constexpr u4 BlockDim = NumShards;
  static constexpr u4 BITWIDTH = sizeof(Hf) * 8;
};

namespace phf {

template <typename Hf = u4>
void make_altcode(Hf* bk, u2 bklen, int reduce_times, Hf& alt_code, u4& alt_bitcount)
{
  using W = HuffmanWord<sizeof(Hf)>;
  auto radius = bklen / 2;
  auto center = bk[radius];
  auto shortest_w = (W*)(&center);

  if (shortest_w->bitcount * reduce_times > (sizeof(Hf) * 8))
    throw std::runtime_error("make alt-code: exceed bits.");

  for (auto i = 0; i < reduce_times; i++) {
    auto offset = W::BITWIDTH - i * shortest_w->bitcount;
    alt_code |= (shortest_w->prefix_code << offset);
  }
  // TODO store bits
}

}  // namespace phf

namespace phf::module {

template <typename T, int Magnitude, int ReduceTimes, bool UseScan = false, typename Hf = u4>
struct HFReVISIT_encode {
  static int CPU_kernel(T* in, const size_t len, phf::book<Hf> book, phf::dense<Hf> dn, void* sp);

  static int GPU_kernel(
      INPUT T* in, const size_t len, phf::book<Hf> book, OUTPUT phf::dense<Hf> dn, void* sp,
      GPU_QUEUE void* stream, DEBUG u4 debug_blockid = false);
};

}  // namespace phf::module

#endif /* AF64552A_247F_47E8_BFFA_EFC88E0491EF */
