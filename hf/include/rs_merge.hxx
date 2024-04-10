#ifndef AF64552A_247F_47E8_BFFA_EFC88E0491EF
#define AF64552A_247F_47E8_BFFA_EFC88E0491EF

#include <stdexcept>

#include "hfcxx_array.hh"
#include "hfword.hh"
#include "mem/memobj.hh"

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

  static constexpr bool DBG = false;
  static constexpr int Magnitude = _Magnitude;
  static constexpr int ReduceTimes = _ReduceTimes;
  static constexpr int ShuffleTimes = Magnitude - ReduceTimes;
  static constexpr int ChunkSize = 1 << Magnitude;
  static constexpr int ShardSize = 1 << ReduceTimes;
  static constexpr int NumShards = 1 << ShuffleTimes;
  static constexpr int BITWIDTH = sizeof(Hf) * 8;
};

namespace phf {
template <typename Hf = u4>
void make_alternative_code(
    memobj<Hf>* bk, int reduce_times, Hf& alt_code, u4& alt_bitcount)
{
  using W = HuffmanWord<sizeof(Hf)>;
  auto radius = bk->len() / 2;
  auto center = bk->hat(radius);
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

namespace phf::cu_hip {

template <
    typename T, int Magnitude, int ReduceTimes, bool use_scan = false,
    typename Hf = u4>
void GPU_HFReVISIT_encode(
    INPUT hfcxx_array<T> in, hfcxx_book<Hf> book,    //
    OUTPUT hfcxx_dense<Hf> dn, hfcxx_compact<T> sp,  //
    GPU_QUEUE void* stream, DEBUG u4 debug_blockid = false);

}

#endif /* AF64552A_247F_47E8_BFFA_EFC88E0491EF */
