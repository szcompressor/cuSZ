#ifndef CUSZ_WRAPPER_HUFFMAN_ENC_DEC_CUH
#define CUSZ_WRAPPER_HUFFMAN_ENC_DEC_CUH

/**
 * @file huffman_enc_dec.cuh
 * @author Jiannan Tian, Cody Rivera (cjrivera1@crimson.ua.edu)
 * @brief Workflow of Huffman coding (header).
 * @version 0.1
 * @date 2020-09-20
 * (created) 2020-04-24 (rev) 2021-09-05
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <tuple>

#include "../capsule.hh"
#include "../type_trait.hh"

using std::string;

/**
 *
 * @tparam UInt unsigned type
 * @param freq frequency
 * @param len data length
 * @param dict_size dictionary size
 * @deprecated merge into Analyzer
 * @return entropy
 */
template <typename UInt>
double get_entropy_from_frequency(UInt* freq, size_t len, size_t dict_size = 1024)
{
    double entropy = 0.0;
    for (auto i = 0; i < dict_size; i++) {
        double prob = freq[i] * 1.0 / len;
        entropy += freq[i] != 0 ? -prob * log2(prob) : 0;
    }
    return entropy;
}

namespace cusz {

template <typename Huff>
void huffman_process_metadata(size_t* _counts, size_t* dev_bits, size_t nchunk, size_t& num_bits, size_t& num_uints);

template <typename Huff>
__global__ void huffman_enc_concatenate(
    Huff*   in_enc_space,
    Huff*   out_bitstream,
    size_t* sp_entries,
    size_t* sp_uints,
    size_t  chunk_size);

}  // namespace cusz

namespace draft {

#ifdef MODULAR_ELSEWHERE
template <typename T>
void UseNvcompZip(T* space, size_t& len);

template <typename T>
void UseNvcompUnzip(T** space, size_t& len);
#endif

}  // namespace draft

namespace lossless {

template <typename Quant, typename Huff, bool UINTS_KNOWN>
void HuffmanEncode(
    Huff*   dev_enc_space,
    size_t* dev_bits,
    size_t* dev_uints,
    size_t* dev_entries,
    size_t* host_counts,
    //
    Huff* dev_out_bitstream,
    //
    Quant*  dev_input,
    Huff*   dev_book,
    size_t  len,
    int     chunk_size,
    int     dict_size,
    size_t* ptr_num_bits,
    size_t* ptr_num_uints,
    float&  milliseconds);

}  // namespace lossless

#endif
