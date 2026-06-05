// Host-callable wrappers for psz::scan_lookback.
// Naming follows CUB (cub/agent/single_pass_scan_operators.cuh).
#ifndef PSZ_SCAN_LOOKBACK_HH
#define PSZ_SCAN_LOOKBACK_HH

#include <cstdint>

namespace psz::scan_lookback {

constexpr int TILE_SIZE_HOST = 1024;

void launch_init_host(
    std::uint32_t* d_partial_aggregate, std::uint32_t* d_incl_prefix, int* d_tile_status,
    int num_tiles, void* stream);

}  // namespace psz::scan_lookback

#endif
