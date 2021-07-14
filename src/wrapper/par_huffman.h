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

// Parallel huffman global memory and kernels
namespace lossless {
namespace par_huffman {

template <typename Q, typename H>
void par_get_codebook(int stateNum, unsigned int* freq, H* codebook, uint8_t* meta);

}  // namespace par_huffman
}  // namespace lossless

#endif
