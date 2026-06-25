// clang-format off

// Warp shuffle intrinsics: HIP's __shfl_*_sync require a 64-bit lane mask, but
// the CUDA-spelled call sites pass a 32-bit mask (0xffffffff). Redirect to the
// maskless __shfl_* forms instead. Variadic so both the 3-arg (default-width)
// and 4-arg (explicit-width) call forms map through.
#define __shfl_sync(MASK, VAR, SRC_LANE, ...) __shfl(VAR, SRC_LANE, ##__VA_ARGS__)
#define __shfl_up_sync(MASK, VAR, DELTA, ...) __shfl_up(VAR, DELTA, ##__VA_ARGS__)
#define __shfl_down_sync(MASK, VAR, DELTA, ...) __shfl_down(VAR, DELTA, ##__VA_ARGS__)
#define __shfl_xor_sync(MASK, VAR, LANE_MASK, ...) __shfl_xor(VAR, LANE_MASK, ##__VA_ARGS__)

// Ballot intrinsic: every call site here uses a logical 32-lane warp (mask
// 0xffffffff) and consumes the result as a 32-bit mask. HIP's __ballot returns
// the full wavefront mask (32 bits on wave32, 64 bits on wave64). On wave64 a
// 32-lane logical warp occupies one of the two 32-lane halves of the wavefront,
// so extract the 32 bits for the current thread's half by shifting down by the
// half-aligned base lane (__lane_id() & ~31). This is correct on both wave32
// (shift is always 0) and wave64, and matches CUDA's uint32_t semantics.
#define __ballot_sync(MASK, PRED) \
  ((unsigned int)(__ballot(PRED) >> (__lane_id() & ~(unsigned)31)))

// Population count and __activemask: HIP uses the same intrinsic name as CUDA
// for __popc; no translation needed.

// clang-format on
