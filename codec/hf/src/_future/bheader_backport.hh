// HF_rev2 per-block header (8 B AoS, bheader-compatible layout).
#ifndef PHF_BHEADER_BACKPORT_HH
#define PHF_BHEADER_BACKPORT_HH

#include <cstdint>

namespace phf {

struct bheader_backport {
  uint32_t bits  : 16;   // par_nbit per partition; sublen ≤ 4096 keeps this < 2^16
  uint32_t entry : 32;   // post-LAGO byte offset (par_entry * sizeof(H))
};

static_assert(sizeof(bheader_backport) == 8, "bheader_backport must be 8 bytes");

}  // namespace phf

#endif  // PHF_BHEADER_BACKPORT_HH
