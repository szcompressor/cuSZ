// Single-threaded Huffman inflate (CPU + GPU).

#ifndef PHF_SINGLE_INFLATE_INL
#define PHF_SINGLE_INFLATE_INL

#include <cstdint>

namespace phf {

// E_storage: in-memory width of the rvbk keys section (defaults to E; set wider
// when the rvbk was built with a wider symbol than the decode target).
// max_out caps emitted symbols; the tail block's total_bw can land mid-codeword
// and otherwise emit one spurious symbol past the block's valid count.
template <typename E, typename H, typename E_storage = E>
__host__ __device__ constexpr void single_thread_inflate(
    H* input, E* out, uint8_t* rvbk, int const total_bw, int const max_out = 0x7fffffff)
{
  constexpr auto H_TYPE_BITS = sizeof(H) * 8;

  int next_bit{};
  auto idx_bit = 0, idx_byte = 0, idx_out = 0;
  H bufr = input[idx_byte];

  auto first = (H*)rvbk;
  auto entry = first + H_TYPE_BITS;
  auto keys = (E_storage*)(rvbk + sizeof(H) * (2 * H_TYPE_BITS));

  H v = (bufr >> (H_TYPE_BITS - 1)) & 0x1;
  auto l = 1, i = 0;

  while (i < total_bw and idx_out < max_out) {
    while (v < first[l]) {
      ++i;
      idx_byte = i / H_TYPE_BITS;
      idx_bit = i % H_TYPE_BITS;
      if (idx_bit == 0) bufr = input[idx_byte];

      next_bit = ((bufr >> (H_TYPE_BITS - 1 - idx_bit)) & 0x1);
      v = (v << 1) | next_bit;
      ++l;
    }
    out[idx_out++] = (E)keys[entry[l] + v - first[l]];

    {
      ++i;
      idx_byte = i / H_TYPE_BITS;
      idx_bit = i % H_TYPE_BITS;
      if (idx_bit == 0) bufr = input[idx_byte];

      next_bit = ((bufr >> (H_TYPE_BITS - 1 - idx_bit)) & 0x1);
      v = 0x0 | next_bit;
    }
    l = 1;
  }
}

}  // namespace phf

#endif  // PHF_SINGLE_INFLATE_INL
