// jtian 20-04-10

#include <cooperative_groups.h>
#include <stddef.h>
#include <stdint.h>
#include "canonical.cuh"

namespace cg = cooperative_groups;

__device__ int max_bw = 0;

// TODO: change H Q order
template <typename H, typename Q>
__global__ void GPU::GetCanonicalCode(uint8_t* singleton, int DICT_SIZE)
{
    auto type_bw   = sizeof(H) * 8;  // this H type takes up (sizeof(H) * 8) bits
    auto codebooks = reinterpret_cast<H*>(singleton);
    auto metadata  = reinterpret_cast<int*>(singleton + sizeof(H) * (3 * DICT_SIZE));
    auto keys      = reinterpret_cast<Q*>(singleton + sizeof(H) * (3 * DICT_SIZE) + sizeof(int) * (4 * type_bw));
    H*   i_cb      = codebooks;                  // specify input-codebook
    H*   o_cb      = codebooks + DICT_SIZE;      // specify output-codebook
    H*   canonical = codebooks + DICT_SIZE * 2;  // specify canonical codes
    auto numl      = metadata;                   // specify numl array
    auto iter_by_  = metadata + type_bw;         // specify intermediate data array
    auto first     = metadata + type_bw * 2;     // specify first-code array
    auto entry     = metadata + type_bw * 3;     // specify (prefix-sum) code array (only sorted by bw)

    cg::grid_group g = cg::this_grid();

    int  gid = blockDim.x * blockIdx.x + threadIdx.x;
    auto c   = i_cb[gid];
    int  bw  = *((uint8_t*)&c + (sizeof(H) - 1));

    if (c != ~((H)0x0)) {
        atomicMax(&max_bw, bw);   // get the maximium bitwidth
        atomicAdd(&numl[bw], 1);  // count numbers of same-bw codes
    }
    g.sync();

    if (gid == 0) {
        memcpy(entry + 1, numl, (type_bw - 1) * sizeof(int));        // TODO: to see par-copy with barrier
        for (int i = 1; i < type_bw; i++) entry[i] += entry[i - 1];  // TODO: prefix-sum, O(n) -> O(log n)
    }
    g.sync();

    if (gid < type_bw) iter_by_[gid] = entry[gid];  // initializing iteration record; will be randomly accessed
    __syncthreads();

    if (gid == 0) {  // sequentially determine first code (RAW)
        for (int l = max_bw - 1; l >= 1; l--) first[l] = static_cast<int>((first[l + 1] + numl[l + 1]) / 2.0 + 0.5);
        first[0] = 0xff;  // in this case, 1-index is used
    }
    g.sync();

    canonical[gid] = ~((H)0x0);  // initialize canonized code
    g.sync();                    // TODO: to remove
    o_cb[gid] = ~((H)0x0);       // initialize output codebook
    g.sync();

    if (gid == 0) {  // RAW (true dependency)
        for (int i = 0; i < DICT_SIZE; i++) {
            auto    _c  = i_cb[i];
            uint8_t _bw = *((uint8_t*)&_c + (sizeof(H) - 1));  // get bw

            if (_c == ~((H)0x0)) continue;  // non-existing codes
            canonical[iter_by_[_bw]] = static_cast<H>(first[_bw] + iter_by_[_bw] - entry[_bw]);
            keys[iter_by_[_bw]]      = i;  // reverse codebook

            *((uint8_t*)&canonical[iter_by_[_bw]] + sizeof(H) - 1) = _bw;  // put bw at MSB
            iter_by_[_bw]++;                                               // update iteration for this bw
        }
    }
    g.sync();

    if (canonical[gid] == ~((H)0x0u)) return;
    o_cb[keys[gid]] = canonical[gid];  // maintaining the orignal order
}

template __global__ void GPU::GetCanonicalCode<uint32_t, uint8_t>(uint8_t* singleton, int DICT_SIZE);
template __global__ void GPU::GetCanonicalCode<uint32_t, uint16_t>(uint8_t* singleton, int DICT_SIZE);
template __global__ void GPU::GetCanonicalCode<uint32_t, uint32_t>(uint8_t* singleton, int DICT_SIZE);
template __global__ void GPU::GetCanonicalCode<uint64_t, uint8_t>(uint8_t* singleton, int DICT_SIZE);
template __global__ void GPU::GetCanonicalCode<uint64_t, uint16_t>(uint8_t* singleton, int DICT_SIZE);
template __global__ void GPU::GetCanonicalCode<uint64_t, uint32_t>(uint8_t* singleton, int DICT_SIZE);
