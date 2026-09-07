#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "hfd26.hh"

namespace {

struct _trace_bounds {
  int start, end;
};

}  // namespace

static _trace_bounds debug_trace_bounds()
{
  const char* s = std::getenv("HFD26_TRACE_START");
  const char* e = std::getenv("HFD26_TRACE_END");
  return {s ? std::atoi(s) : -1, e ? std::atoi(e) : -1};
}

namespace phf::cpu_ref {

template <typename H, typename KStorage>
void build_lut(uint8_t const* rvbk, LutEntry* out_lut)
{
  constexpr auto H_BITS = sizeof(H) * 8;
  auto first = reinterpret_cast<H const*>(rvbk);
  auto entry = first + H_BITS;
  auto keys = reinterpret_cast<KStorage const*>(rvbk + sizeof(H) * (2 * H_BITS));

  for (auto p = 0; p < 256; ++p) {
    LutEntry e = {0, 0, 0};
    for (auto L = 1; L <= 8; ++L) {
      H v = (H)((unsigned)p >> (8 - L));
      H first_L = first[L];
      H count_L = entry[L + 1] - entry[L];
      if (v >= first_L && v < first_L + count_L) {
        e.symbol = (uint16_t)keys[entry[L] + v - first_L];
        e.length = (uint8_t)L;
        break;
      }
    }
    out_lut[p] = e;
  }
}

template <typename E, typename H, typename KStorage>
void shard_inflate_lut(
    H const* bs_base, int bit_start, int bit_end, LutEntry const* lut, uint8_t const* rvbk, E* out,
    int max_out)
{
  constexpr auto H_BITS = sizeof(H) * 8;
  if (bit_end <= bit_start) return;

  auto first = reinterpret_cast<H const*>(rvbk);
  auto entry = first + H_BITS;
  auto keys = reinterpret_cast<KStorage const*>(rvbk + sizeof(H) * (2 * H_BITS));

  auto i = bit_start;
  auto idx_out = 0;
  auto tb = debug_trace_bounds();

  while (i < bit_end and idx_out < max_out) {
    if (bit_end - i >= 8) {
      // fast path: peek 8 bits at position i, LUT lookup.
      auto const word_idx = i >> 5;
      auto const word_bit = i & 31;
      H w0 = bs_base[word_idx];
      H w1 = (word_bit > 24) ? bs_base[word_idx + 1] : (H)0;
      H combined =
          (w0 << word_bit) | ((word_bit == 0) ? (H)0 : (w1 >> (H_BITS - word_bit)));
      unsigned peek = (unsigned)((combined >> 24) & 0xFFu);

      LutEntry e = lut[peek];
      if (idx_out >= tb.start && idx_out < tb.end) {
        std::fprintf(
            stderr,
            "  [%d] i=%d  word_idx=%d word_bit=%d  w0=0x%08x  peek=0x%02x  LUT.length=%u "
            "LUT.sym=%u\n",
            idx_out, i, word_idx, word_bit, (unsigned)w0, peek, (unsigned)e.length,
            (unsigned)e.symbol);
      }
      if (e.length > 0) {
        out[idx_out++] = (E)e.symbol;
        i += (int)e.length;
        continue;
      }

      // slow path: codeword is 9-16 bits.
      H v = (H)peek;
      auto L = 8;
      while (L < 16) {
        ++L;
        auto const bit_idx = i + L - 1;
        auto const wi = bit_idx >> 5;
        auto const bi = bit_idx & 31;
        H next = (bs_base[wi] >> (H_BITS - 1 - bi)) & (H)0x1u;
        v = (H)((v << 1) | next);
        H first_L = first[L];
        H count_L = entry[L + 1] - entry[L];
        if (v >= first_L && v < first_L + count_L) break;
      }
      if (idx_out >= tb.start && idx_out < tb.end)
        std::fprintf(
            stderr, "  [%d] i=%d SLOW peek=0x%02x L=%d v=0x%x first[L]=0x%x cnt=0x%x sym=%u\n",
            idx_out, i, peek, L, (unsigned)v, (unsigned)first[L],
            (unsigned)(entry[L + 1] - entry[L]), (unsigned)keys[entry[L] + v - first[L]]);
      out[idx_out++] = (E)keys[entry[L] + v - first[L]];
      i += L;
    }
    else {
      // tail: < 8 bits, bit-walker.
      auto idx_word = i >> 5;
      auto idx_bit = i & 31;
      H bufr = bs_base[idx_word];
      H v = (bufr >> (H_BITS - 1 - idx_bit)) & (H)0x1u;
      auto l = 1;
      while (v < first[l]) {
        ++i;
        idx_word = i >> 5;
        idx_bit = i & 31;
        if (idx_bit == 0) bufr = bs_base[idx_word];
        H next = (bufr >> (H_BITS - 1 - idx_bit)) & (H)0x1u;
        v = (H)((v << 1) | next);
        ++l;
      }
      out[idx_out++] = (E)keys[entry[l] + v - first[l]];
      ++i;
    }
  }
}

template <typename H>
int walk_n(H const* bs, H const* first, int i, int bit_end, int count)
{
  constexpr auto H_BITS = sizeof(H) * 8;
  while (count > 0 and i < bit_end) {
    auto idx_word = i >> 5, idx_bit = i & 31;
    H bufr = bs[idx_word];
    H v = (bufr >> (H_BITS - 1 - idx_bit)) & (H)0x1u;
    auto l = 1;
    while (v < first[l]) {
      ++i;
      idx_word = i >> 5;
      idx_bit = i & 31;
      if (idx_bit == 0) bufr = bs[idx_word];
      H next = (bufr >> (H_BITS - 1 - idx_bit)) & (H)0x1u;
      v = (H)((v << 1) | next);
      ++l;
      if (l > H_BITS) return i;
    }
    ++i;
    --count;
  }
  return i;
}

// instantiation
template void build_lut<uint32_t, uint8_t>(uint8_t const*, LutEntry*);
template void build_lut<uint32_t, uint16_t>(uint8_t const*, LutEntry*);
template void shard_inflate_lut<uint8_t, uint32_t, uint8_t>(
    uint32_t const*, int, int, LutEntry const*, uint8_t const*, uint8_t*, int);
template void shard_inflate_lut<uint16_t, uint32_t, uint8_t>(
    uint32_t const*, int, int, LutEntry const*, uint8_t const*, uint16_t*, int);
template void shard_inflate_lut<uint16_t, uint32_t, uint16_t>(
    uint32_t const*, int, int, LutEntry const*, uint8_t const*, uint16_t*, int);
template int walk_n<uint32_t>(uint32_t const*, uint32_t const*, int, int, int);

}  // namespace phf::cpu_ref
