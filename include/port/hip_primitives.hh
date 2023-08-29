// clang-format off
#define __shfl_sync(MASK, VAR, SRC_LANE, WIDTH) __shfl(VAR, SRC_LANE, WIDTH)
#define __shfl_up_sync(MASK, VAR, DELTA, WIDTH) __shfl_up(VAR, DELTA, WIDTH)
#define __shfl_down_sync(MASK, VAR, DELTA, WIDTH) __shfl_down(VAR, DELTA, WIDTH)
#define __shfl_xor_sync(MASK, VAR, LANE_MASK, WIDTH) __shfl_xor(VAR, LANE_MASK, WIDTH)
// clang-format on