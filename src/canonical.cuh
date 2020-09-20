#ifndef CANONICAL_CUH
#define CANONICAL_CUH

/**
 * @file canonical.cuh
 * @author Jiannan Tian
 * @brief Canonization of existing Huffman codebook (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-04-10
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cstdint>

namespace GPU {

__device__ int max_bw;

template <typename T, typename K>
__global__ void GetCanonicalCode(uint8_t* singleton, int DICT_SIZE);

}  // namespace GPU
#endif
