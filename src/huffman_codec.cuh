#ifndef DEFLATE_CUH
#define DEFLATE_CUH

#include <stddef.h>

template <typename Q, typename H>
__global__ void EncodeFixedLen(Q* data, H* hcoded, size_t data_len, H* codebook);

template <typename Q>
__global__ void Deflate(Q* hcoded, size_t len, size_t* densely_meta, int PART_SIZE);

template <typename H, typename T>
__device__ void InflateChunkwise(H* in_dHcoded, T* out_bcoded, size_t total_bw, uint8_t* singleton);

template <typename Q, typename H>
__global__ void Decode(H* densely, size_t* dH_meta, Q* bcode, size_t len, int chunk_size, int n_chunk, uint8_t* singleton, size_t singleton_size);

#endif
