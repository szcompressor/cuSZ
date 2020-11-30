#ifndef DEFLATE_CUH
#define DEFLATE_CUH

/**
 * @file huffman_codec.cu
 * @author Jiannan Tian
 * @brief Wrapper of Huffman codec (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-02-02
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <stddef.h>

template <typename Q, typename H>
__global__ void EncodeFixedLen(Q*, H*, size_t, H*);

template <typename Q>
__global__ void Deflate(Q*, size_t, size_t*, int);

template <typename H, typename T>
__device__ void InflateChunkwise(H*, T*, size_t, uint8_t*);

template <typename Q, typename H>
__global__ void Decode(H*, size_t*, Q*, size_t, int, int, uint8_t*, size_t);

#endif
