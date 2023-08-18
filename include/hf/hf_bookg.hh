/**
 * @file huffman_parbook.cuh
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

#include <stdint.h>

void lambda_sort_by_freq(uint32_t* freq, const int len, uint32_t* qcode);

namespace par_huffman {
// Codeword length
template <typename F>
__global__ void GPU_GenerateCL(F*, F*, int, F*, int*, F*, int*, F*, int*, int*, F*, int*, int*, uint32_t*, int, int);

// Forward Codebook
template <typename F, typename H>
__global__ void GPU_GenerateCW(F* CL, H* CW, H* first, H* entry, int size);

}  // namespace par_huffman

// Parallel huffman global memory and kernels
namespace psz {

template <typename T, typename H>
void hf_buildbook_cu(
    uint32_t* freq,
    int const bklen,
    H*        book,
    uint8_t*  revbook,
    int const revbook_nbyte,
    float*    time,
    cudaStream_t = nullptr);

}  // namespace psz

#endif
