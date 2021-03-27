#ifndef HUFFMAN_WORKFLOW
#define HUFFMAN_WORKFLOW

/**
 * @file huffman_workflow.cuh
 * @author Jiannan Tian, Cody Rivera (cjrivera1@crimson.ua.edu)
 * @brief Workflow of Huffman coding (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-04-24
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cuda_runtime.h>
//#include <sys/stat.h>

#include <cstdint>
#include <string>
#include <tuple>

#include "type_trait.hh"
#include "datapack.hh"

using std::string;

typedef std::tuple<size_t, size_t, size_t, bool> tuple3ul;

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
double GetEntropyFromFrequency(UInt* freq, size_t len, size_t dict_size = 1024)
{
    double entropy = 0.0;
    for (auto i = 0; i < dict_size; i++) {
        double prob = freq[i] * 1.0 / len;
        entropy += freq[i] != 0 ? -prob * log2(prob) : 0;
    }
    return entropy;
}

namespace draft {

template <typename Huff>
void GatherSpHuffMetadata(size_t* _counts, size_t* d_sp_bits, size_t nchunk, size_t& total_bits, size_t& total_uints);

template <typename Huff>
__global__ void CopyHuffmanUintsDenseToSparse(Huff*, Huff*, size_t*, size_t*, size_t);

template <typename T>
void UseNvcompZip(T* space, size_t& len);

template <typename T>
void UseNvcompUnzip(T** space, size_t& len);

}  // namespace draft

namespace lossless {

namespace wrapper {
template <typename Input>
void GetFrequency(Input*, size_t, unsigned int*, int);

}  // namespace wrapper

namespace interface {

template <typename Quant, typename Huff, typename Data = float>
std::tuple<size_t, size_t, size_t, bool>
HuffmanEncode(string&, Quant*, Huff*, uint8_t*, size_t, size_t, int, bool, int dict_size = 1024);

template <typename Quant, typename Huff, typename Data = float>
void HuffmanDecode(std::string&, DataPack<Quant>*, size_t, int, size_t, bool, int dict_size = 1024);

template <typename Quant, typename Huff, typename Data = float>
void HuffmanEncodeWithTree_3D(Index<3>::idx_t idx, string& basename, Quant* h_quant_in, size_t len, int dict_size);

}  // namespace interface
}  // namespace lossless

#endif
