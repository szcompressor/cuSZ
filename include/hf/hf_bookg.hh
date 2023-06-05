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
namespace psz {

/**
 * @brief get codebook and reverse codebook in parallel
 *
 * @tparam T input type
 * @tparam H codebook type
 * @param freq input device array; frequency
 * @param codebook output device array; codebook for encoding
 * @param dict_size dictionary size; len of freq or codebook
 * @param reverse_codebook output device array; reverse codebook for decoding
 * @param _time_book the returned time
 */
template <typename T, typename H>
void hf_buildbook_g(
    uint32_t* freq,
    int const booksize,
    H*        codebook,
    uint8_t*  reverse_codebook,
    int const revbook_nbyte,
    float*    _time_book,
    cudaStream_t = nullptr);

}  // namespace psz

#endif
