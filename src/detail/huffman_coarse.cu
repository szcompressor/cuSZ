/**
 * @file huffman_coarse.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-12-17
 * (created) 2020-04-24 (rev1) 2021-09-05 (rev2) 2021-12-29
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * @copyright (C) 2021 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include "detail/huffman_coarse.cuh"

#define HUFFCOARSE(E, ETF, H, M) \
    template class cusz::HuffmanCoarse<ErrCtrlTrait<E, ETF>::type, HuffTrait<H>::type, MetadataTrait<M>::type>::impl;

HUFFCOARSE(2, false, 4, 4)  // deprecated
HUFFCOARSE(2, false, 8, 4)  // deprecated
HUFFCOARSE(4, false, 4, 4)  // deprecated
HUFFCOARSE(4, false, 8, 4)  // deprecated

HUFFCOARSE(4, true, 4, 4)  // float
HUFFCOARSE(4, true, 8, 4)  // float

#undef HUFFCOARSE
