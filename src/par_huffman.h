/**
 * @file par_huffman.cuh
 * @author Cody Rivera (cjrivera1@crimson.ua.edu)
 * @brief Parallel Huffman Construction to generates canonical forward codebook (header).
 *        Based on [Ostadzadeh et al. 2007] (https://dblp.org/rec/conf/pdpta/OstadzadehEZMB07.bib)
 *        "A Two-phase Practical Parallel Algorithm for Construction of Huffman Codes".
 * @version 0.1
 * @date 2020-09-20
 * Created on: 2020-06
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef PAR_HUFFMAN_H
#define PAR_HUFFMAN_H

#include <cooperative_groups.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace std;
using namespace cooperative_groups;

// Parallel huffman global memory and kernels
namespace lossless {
namespace par_huffman {
namespace helper {

// clang-format off
template <typename T>             __global__ void GPU_FillArraySequence(T*, unsigned int);
template <typename T>             __global__ void GPU_GetFirstNonzeroIndex(T*, unsigned int, unsigned int*);
template <typename T>             __global__ void GPU_ReverseArray(T*, unsigned int);
template <typename T, typename Q> __global__ void GPU_ReorderByIndex(T*, Q*, unsigned int);
// clang-format on

}  // namespace helper

// Codeword length
template <typename F>
__global__ void GPU_GenerateCL(F*, F*, int, F*, int*, F*, int*, F*, int*, int*, F*, int*, int*, uint32_t*, int, int);

// Forward Codebook
template <typename F, typename H>
__global__ void GPU_GenerateCW(F* CL, H* CW, H* first, H* entry, int size);

template <typename Q, typename H>
void ParGetCodebook(int stateNum, unsigned int* freq, H* codebook, uint8_t* meta);

}  // namespace par_huffman
}  // namespace lossless

#endif
