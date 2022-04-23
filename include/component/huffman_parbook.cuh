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

// Parallel huffman global memory and kernels
namespace kernel_wrapper {

/**
 * @brief get codebook and reverse codebook in parallel
 *
 * @tparam T input type
 * @tparam H codebook type
 * @param freq input device array; frequency
 * @param dict_size dictionary size; len of freq
 * @param codebook output device array; codebook for encoding
 * @param reverse_codebook output device array; reverse codebook for decoding
 */
template <typename T, typename H>
void par_get_codebook(cusz::FREQ* freq, int dict_size, H* codebook, uint8_t* reverse_codebook, cudaStream_t = nullptr);

}  // namespace kernel_wrapper

#endif
