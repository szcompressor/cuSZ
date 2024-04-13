/**
 * @file canonical.cu
 * @author Jiannan Tian
 * @brief Canonization of existing Huffman codebook.
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-04-10
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory See LICENSE in top-level directory
 *
 */

#include <cooperative_groups.h>
#include <stddef.h>
#include <stdint.h>

#include <cstdint>

#include "hf/hf_canon.hh"


__device__ int max_bw = 0;

namespace cg = cooperative_groups;


template <typename E, typename H>
__global__ void hf_canonical_code_cuda_cg(uint8_t* singleton, uint32_t booklen)
{
  auto TYPE_BITS = sizeof(H) * 8;
  auto codebooks = reinterpret_cast<H*>(singleton);
  auto metadata =
      reinterpret_cast<int*>(singleton + sizeof(H) * (3 * booklen));
  auto keys = reinterpret_cast<E*>(
      singleton + sizeof(H) * (3 * booklen) + sizeof(int) * (4 * TYPE_BITS));
  H* i_cb = codebooks;
  H* o_cb = codebooks + booklen;
  H* canonical = codebooks + booklen * 2;
  auto numl = metadata;
  auto iter_by_ = metadata + TYPE_BITS;
  auto first = metadata + TYPE_BITS * 2;
  auto entry = metadata + TYPE_BITS * 3;

  cg::grid_group g = cg::this_grid();

  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  // TODO
  auto c = i_cb[gid];
  int bw = *((uint8_t*)&c + (sizeof(H) - 1));

  if (c != ~((H)0x0)) {
    atomicMax(&max_bw, bw);
    atomicAdd(&numl[bw], 1);
  }
  g.sync();

  if (gid == 0) {
    // printf("\0");
    // atomicMax(&max_bw, max_bw + 0);
    memcpy(entry + 1, numl, (TYPE_BITS - 1) * sizeof(int));
    // for (int i = 1; i < TYPE_BITS; i++) entry[i] = numl[i - 1];
    for (int i = 1; i < TYPE_BITS; i++) entry[i] += entry[i - 1];
  }
  g.sync();

  if (gid < TYPE_BITS) iter_by_[gid] = entry[gid];
  __syncthreads();
  // atomicMax(&max_bw, bw);

  if (gid == 0) {  //////// first code
    for (int l = max_bw - 1; l >= 1; l--)
      first[l] = static_cast<int>((first[l + 1] + numl[l + 1]) / 2.0 + 0.5);
    first[0] = 0xff;  // no off-by-one error
  }
  g.sync();

  canonical[gid] = ~((H)0x0);
  g.sync();
  o_cb[gid] = ~((H)0x0);
  g.sync();

  // Reverse Codebook Generation -- TODO isolate
  if (gid == 0) {
    // no atomicRead to handle read-after-write (true dependency)
    for (int i = 0; i < booklen; i++) {
      auto _c = i_cb[i];
      uint8_t _bw = *((uint8_t*)&_c + (sizeof(H) - 1));

      if (_c == ~((H)0x0)) continue;
      canonical[iter_by_[_bw]] =
          static_cast<H>(first[_bw] + iter_by_[_bw] - entry[_bw]);
      keys[iter_by_[_bw]] = i;

      *((uint8_t*)&canonical[iter_by_[_bw]] + sizeof(H) - 1) = _bw;
      iter_by_[_bw]++;
    }
  }
  g.sync();

  if (canonical[gid] == ~((H)0x0u)) return;
  o_cb[keys[gid]] = canonical[gid];
}


template <typename E, typename H>
__global__ void canonize_cuda(uint8_t* singleton, uint32_t booklen)
{
  auto TYPE_BITS = sizeof(H) * 8;
  auto codebooks = reinterpret_cast<H*>(singleton);
  auto metadata =
      reinterpret_cast<int*>(singleton + sizeof(H) * (3 * booklen));
  auto keys = reinterpret_cast<E*>(
      singleton + sizeof(H) * (3 * booklen) + sizeof(int) * (4 * TYPE_BITS));
  H* i_cb = codebooks;
  H* o_cb = codebooks + booklen;
  H* canonical = codebooks + booklen * 2;
  auto numl = metadata;
  auto iter_by_ = metadata + TYPE_BITS;
  auto first = metadata + TYPE_BITS * 2;
  auto entry = metadata + TYPE_BITS * 3;

  __shared__ int max_bw;

  int gid = threadIdx.x;
  auto c = i_cb[gid];
  int bw = *((uint8_t*)&c + (sizeof(H) - 1));

  if (c != ~((H)0x0)) {
    atomicMax(&max_bw, bw);
    atomicAdd(&numl[bw], 1);
  }
  __syncthreads();

  if (gid == 0) {
    memcpy(entry + 1, numl, (TYPE_BITS - 1) * sizeof(int));
    for (int i = 1; i < TYPE_BITS; i++) entry[i] += entry[i - 1];
  }
  __syncthreads();

  if (gid < TYPE_BITS) iter_by_[gid] = entry[gid];
  __syncthreads();

  if (gid == 0) {  //////// first code
    for (int l = max_bw - 1; l >= 1; l--)
      first[l] = static_cast<int>((first[l + 1] + numl[l + 1]) / 2.0 + 0.5);
    first[0] = 0xff;  // no off-by-one error
  }
  __syncthreads();

  canonical[gid] = ~((H)0x0);
  __syncthreads();
  o_cb[gid] = ~((H)0x0);
  __syncthreads();

  // Reverse Codebook Generation -- TODO isolate
  if (gid == 0) {
    // no atomicRead to handle read-after-write (true dependency)
    for (int i = 0; i < booklen; i++) {
      auto _c = i_cb[i];
      uint8_t _bw = *((uint8_t*)&_c + (sizeof(H) - 1));

      if (_c == ~((H)0x0)) continue;
      canonical[iter_by_[_bw]] =
          static_cast<H>(first[_bw] + iter_by_[_bw] - entry[_bw]);
      keys[iter_by_[_bw]] = i;

      *((uint8_t*)&canonical[iter_by_[_bw]] + sizeof(H) - 1) = _bw;
      iter_by_[_bw]++;
    }
  }
  __syncthreads();

  if (canonical[gid] == ~((H)0x0u)) return;
  o_cb[keys[gid]] = canonical[gid];
}

#define SPECIALIZE(E, H)                                               \
  template <>                                                          \
  int canonize_on_gpu<E, H>(                                           \
      uint8_t * binary_in, uint32_t booklen, void* stream)             \
  {                                                                    \
    canonize_cuda<E, H>                                                \
        <<<1, booklen, 0, (cudaStream_t)stream>>>(binary_in, booklen); \
    cudaStreamSynchronize((cudaStream_t)stream);                       \
    return 0;                                                          \
  }

SPECIALIZE(uint8_t, uint32_t)
SPECIALIZE(uint16_t, uint32_t)
SPECIALIZE(uint32_t, uint32_t)
SPECIALIZE(uint8_t, uint64_t)
SPECIALIZE(uint16_t, uint64_t)
SPECIALIZE(uint32_t, uint64_t)

#undef SPECIALIZE
