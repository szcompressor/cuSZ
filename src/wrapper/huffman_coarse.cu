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

#include "huffman_coarse.cuh"

#define HUFFCOARSE(E, H, M) \
    template class cusz::HuffmanCoarse<ErrCtrlTrait<E>::type, HuffTrait<H>::type, MetadataTrait<M>::type>;

HUFFCOARSE(2, 4, 4)
// HUFFCOARSE(2, 4, 8)
HUFFCOARSE(2, 8, 4)
// HUFFCOARSE(2, 8, 8)

// HUFFCOARSE(4, 4, 4)
// HUFFCOARSE(4, 4, 8)
// HUFFCOARSE(4, 8, 4)
// HUFFCOARSE(4, 8, 8)

#undef HUFFCOARSE
