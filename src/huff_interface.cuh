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

using std::string;

// const int GB_unit = 1073741824;  // 1024^3

const int tBLK_ENCODE    = 256;
const int tBLK_DEFLATE   = 128;
const int tBLK_CANONICAL = 128;

// https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
// inline bool exists_test2(const std::string& name) {
//    return (access(name.c_str(), F_OK) != -1);
//}

typedef std::tuple<size_t, size_t, size_t, bool> tuple3ul;

// clang-format off
namespace lossless {

namespace wrapper {
template <typename UInt_Input> void GetFrequency(UInt_Input*, size_t, unsigned int*, int);
}  // namespace wrapper

namespace utils {
template <typename H> void PrintChunkHuffmanCoding(size_t*, size_t*, size_t, int, size_t, size_t);
}

namespace interface {

template <typename Quant, typename Huff, typename Data = float>
tuple3ul HuffmanEncode(string& basename, Quant* d_in, size_t len, int chunk_size, bool to_nvcomp, int dict_size = 1024);

template <typename Quant, typename Huff, typename Data = float>
Quant* HuffmanDecode(std::string& basename, size_t len, int chunk_size, int total_uInts, bool nvcomp_in_use, int dict_size = 1024);

}  // namespace interface
}  // namespace lossless

// clang-format on

#endif
