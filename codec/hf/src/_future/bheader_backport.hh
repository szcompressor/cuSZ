// HF_rev2 per-block header (8 B AoS, bheader-compatible layout).
#ifndef PHF_BHEADER_BACKPORT_HH
#define PHF_BHEADER_BACKPORT_HH

#include <cstdint>

namespace phf {

struct bheader_backport {
  uint32_t bits;    // par_nbit per partition (full 32-bit; sublen*codeword_len can exceed 2^16)
  uint32_t entry;   // post-LAGO byte offset (par_entry * sizeof(H))
};

static_assert(sizeof(bheader_backport) == 8, "bheader_backport must be 8 bytes");

}  // namespace phf

#endif  // PHF_BHEADER_BACKPORT_HH
