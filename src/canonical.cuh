#ifndef CANONICAL_CUH
#define CANONICAL_CUH

//#include <algorithm>
#include <cooperative_groups.h>
#include <cstddef>
#include <iostream>

namespace cg = cooperative_groups;
// using namespace cooperative_groups;

namespace GPU {

template <typename H, typename T>
__device__ void Inflate_v2(H* in_dHcoded, T* out_bcoded, size_t total_bw, uint8_t* singleton) {
    uint8_t   next_bit;
    size_t    idx_bit;
    size_t    idx_byte   = 0;
    size_t    idx_bcoded = 0;
    auto      first      = reinterpret_cast<int*>(singleton);
    auto      entry      = first + sizeof(H) * 8;
    auto      keys       = reinterpret_cast<uint16_t*>(singleton + sizeof(int) * (2 * sizeof(H) * 8));
    ptrdiff_t v          = (in_dHcoded[idx_byte] >> (sizeof(H) * 8 - 1)) & 0x1;  // get the first bit
    size_t    l          = 1;
    size_t    i          = 0;
    while (i < total_bw) {
        while (v < first[l]) {  // append next i_cb bit
            ++i;
            idx_byte = i / (sizeof(H) * 8);
            idx_bit  = i % (sizeof(H) * 8);

            next_bit = ((in_dHcoded[idx_byte] >> (sizeof(H) * 8 - 1 - idx_bit)) & 0x1);
            v        = (v << 1) | next_bit;
            l++;
        }
        out_bcoded[idx_bcoded++] = keys[entry[l] + v - first[l]];
        {
            ++i;
            idx_byte = i / (sizeof(H) * 8);
            idx_bit  = i % (sizeof(H) * 8);
            next_bit = ((in_dHcoded[idx_byte] >> (sizeof(H) * 8 - 1 - idx_bit)) & 0x1);
            v        = 0x0 | next_bit;
        }
        l = 1;
    }
}

template <typename Q, typename T>
__global__ void inflate(Q* densely, T* bcode, size_t total_bw, uint8_t* singleton) {
    uint8_t   next_bit;
    size_t    idx_bit;
    size_t    idx_byte  = 0;
    size_t    idx_bcode = 0;
    auto      first     = reinterpret_cast<int*>(singleton);
    auto      entry     = first + sizeof(Q) * 8;
    auto      keys      = reinterpret_cast<uint16_t*>(singleton + sizeof(int) * (2 * sizeof(Q) * 8));
    ptrdiff_t v         = (densely[idx_byte] >> (sizeof(Q) * 8 - 1)) & 0x1;  // get the first bit
    size_t    l         = 1;
    size_t    i         = 0;
    while (i < total_bw) {
        while (v < first[l]) {  // append next i_cb bit
            ++i;
            idx_byte = i / (sizeof(Q) * 8);
            idx_bit  = i % (sizeof(Q) * 8);

            next_bit = ((densely[idx_byte] >> (sizeof(Q) * 8 - 1 - idx_bit)) & 0x1);
            v        = (v << 1) | next_bit;
            l++;
        }
        bcode[idx_bcode++] = keys[entry[l] + v - first[l]];
        {
            ++i;
            idx_byte = i / (sizeof(Q) * 8);
            idx_bit  = i % (sizeof(Q) * 8);
            next_bit = ((densely[idx_byte] >> (sizeof(Q) * 8 - 1 - idx_bit)) & 0x1);
            v        = 0x0 | next_bit;
        }
        l = 1;
    }
}

// TODO auto change to 64 (e.g., exafel 1e-4)
__device__ int max_bw = 0;

template <typename T, typename K>
__global__ void GetCanonicalCode(uint8_t* singleton, int DICT_SIZE) {
    auto type_bw   = sizeof(T) * 8;
    auto codebooks = reinterpret_cast<T*>(singleton);
    auto metadata  = reinterpret_cast<int*>(singleton + sizeof(T) * (3 * DICT_SIZE));
    auto keys      = reinterpret_cast<K*>(singleton + sizeof(T) * (3 * DICT_SIZE) + sizeof(int) * (4 * type_bw));
    T*   i_cb      = codebooks;
    T*   o_cb      = codebooks + DICT_SIZE;
    T*   canonical = codebooks + DICT_SIZE * 2;
    auto numl      = metadata;
    auto iter_by_  = metadata + type_bw;
    auto first     = metadata + type_bw * 2;
    auto entry     = metadata + type_bw * 3;

    cg::grid_group g = cg::this_grid();

    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    // TODO
    auto c  = i_cb[gid];
    int  bw = *((uint8_t*)&c + (sizeof(T) - 1));

    if (c != ~((T)0x0)) {
        atomicMax(&max_bw, bw);
        atomicAdd(&numl[bw], 1);
    }
    g.sync();

    if (gid == 0) {
        // printf("\0");
        // atomicMax(&max_bw, max_bw + 0);
        memcpy(entry + 1, numl, (type_bw - 1) * sizeof(int));
        // for (int i = 1; i < type_bw; i++) entry[i] = numl[i - 1];
        for (int i = 1; i < type_bw; i++) entry[i] += entry[i - 1];
    }
    g.sync();

    if (gid < type_bw) iter_by_[gid] = entry[gid];
    __syncthreads();
    // atomicMax(&max_bw, bw);

    if (gid == 0) {  //////// first code
        for (int l = max_bw - 1; l >= 1; l--) first[l] = static_cast<int>((first[l + 1] + numl[l + 1]) / 2.0 + 0.5);
        first[0] = 0xff;  // no off-by-one error
    }
    g.sync();

    canonical[gid] = ~((T)0x0);
    g.sync();
    o_cb[gid] = ~((T)0x0);
    g.sync();

    if (gid == 0) {
        // no atomicRead to handle read-after-write (true dependency)
        for (int i = 0; i < DICT_SIZE; i++) {
            auto    _c  = i_cb[i];
            uint8_t _bw = *((uint8_t*)&_c + (sizeof(T) - 1));

            if (_c == ~((T)0x0)) continue;
            canonical[iter_by_[_bw]] = static_cast<T>(first[_bw] + iter_by_[_bw] - entry[_bw]);
            keys[iter_by_[_bw]]      = i;

            *((uint8_t*)&canonical[iter_by_[_bw]] + sizeof(T) - 1) = _bw;
            iter_by_[_bw]++;
        }
    }
    g.sync();

    if (canonical[gid] == ~((T)0x0u)) return;
    o_cb[keys[gid]] = canonical[gid];
}

}  // namespace GPU
#endif
