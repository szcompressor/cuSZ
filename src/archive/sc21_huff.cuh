/**
 * @file huffman_workflow.cu
 * @author Jiannan Tian, Cody Rivera (cjrivera1@crimson.ua.edu)
 * @brief Workflow of Huffman coding.
 * @version 0.1
 * @date 2020-10-24
 * Created on 2020-04-24
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "hist.cuh"
#include "huff_codec.cuh"
#include "type_aliasing.hh"
#include "type_trait.hh"
#include "types.hh"
#include "utils/cuda_err.cuh"
#include "utils/cuda_mem.cuh"
#include "utils/format.hh"
#include "utils/io.hh"
#include "utils/timer.hh"

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

struct SC21HuffmanHeader {
    unsigned int this_header_size;
    unsigned int chunk_size;
    unsigned int num_chunks;
    size_t       num_bits;
    size_t       num_units;

    struct {
        size_t reverse_book;
        size_t of_chunks;
        size_t compact_huff;
    } len_in_byte;

    struct {
        size_t reverse_book;
        size_t of_chunks;
        size_t compact_huff;
    } loc;
};

#define nworker blockDim.x

namespace sc21 {

template <typename Huff>
__global__ void GatherEncodedChunks(
    Huff*   input_dense,
    Huff*   output_sparse,
    size_t* entries_of_chunks,
    size_t* units_of_chunks,
    size_t  chunk_size)
{
    auto len      = units_of_chunks[blockIdx.x];
    auto sp_entry = entries_of_chunks[blockIdx.x];
    auto dn_entry = chunk_size * blockIdx.x;

    for (auto i = 0; i < (len + nworker - 1) / nworker; i++) {
        auto _tid = threadIdx.x + i * nworker;
        if (_tid < len) *(output_sparse + sp_entry + _tid) = *(input_dense + dn_entry + _tid);
        __syncthreads();
    }
}

template <typename Huff>
void ProcessHuffmanMetadataAfterChunkwiseEncoding(
    size_t* h__uninitialized,
    size_t* d__bits_of_chunks,
    size_t  num_chunks,
    size_t& num_bits,
    size_t& num_units)
{
    static const size_t Huff_bytes = sizeof(Huff) * 8;

    // TODO *_of_chunks -> chunkwise_*
    auto units_of_chunks   = h__uninitialized;
    auto bits_of_chunks    = h__uninitialized + num_chunks;
    auto entries_of_chunks = h__uninitialized + num_chunks * 2;

    cudaMemcpy(bits_of_chunks, d__bits_of_chunks, num_chunks * sizeof(size_t), cudaMemcpyDeviceToHost);
    memcpy(units_of_chunks, bits_of_chunks, num_chunks * sizeof(size_t));
    for_each(units_of_chunks, units_of_chunks + num_chunks, [&](size_t& i) { i = (i + Huff_bytes - 1) / Huff_bytes; });
    memcpy(entries_of_chunks + 1, units_of_chunks, (num_chunks - 1) * sizeof(size_t));
    for (auto i = 1; i < num_chunks; i++) entries_of_chunks[i] += entries_of_chunks[i - 1];  // inclusive scan

    num_bits  = std::accumulate(bits_of_chunks, bits_of_chunks + num_chunks, (size_t)0);
    num_units = std::accumulate(units_of_chunks, units_of_chunks + num_chunks, (size_t)0);
}

template <typename Quant, typename Huff>
void lossless::interface::HuffmanEncode(
    string&  basename,
    Quant*   d__input,
    size_t   len,
    Huff*    d__book,
    int      book_len,
    uint8_t* d__reverse_book,
    size_t   nbyte_reverse_book,
    int      chunk_size,
    size_t&  num_bits,   // output
    size_t&  num_units,  // output
    // size_t&  metadata_size  // output
)
{
    static const auto type_bitcount = sizeof(Huff) * 8;  // canonical Huffman; follows H to decide first and entry type

    auto get_Dg = [](size_t problem_size, size_t Db) { return (problem_size + Db - 1) / Db; };

    // auto h__reverse_book = mem::create_devspace_memcpy_d2h(d__reverse_book, nbyte_reverse_book);
    // io::write_array_to_binary(
    //     basename + ".canon", reinterpret_cast<uint8_t*>(h__reverse_book),
    //     sizeof(Huff) * (2 * type_bitcount) + sizeof(Quant) * book_len);
    // delete[] h__reverse_book;

    // Huffman space in dense format (full of zeros), fix-length space; TODO ad hoc (big) padding
    auto d__enc_space = mem::create_CUDA_space<Huff>(len + chunk_size + HuffConfig::Db_encode);
    {
        auto Db = HuffConfig::Db_encode;
        lossless::wrapper::encode_fixedlen_space_cub<Quant, Huff, HuffConfig::enc_sequentiality>
            <<<get_Dg(len, Db), Db / HuffConfig::enc_sequentiality>>>(d__input, d__enc_space, len, d__book);
        cudaDeviceSynchronize();
    }

    // encode_deflate
    auto num_chunks        = (len + chunk_size - 1) / chunk_size;
    auto d__bits_of_chunks = mem::create_CUDA_space<size_t>(num_chunks);
    {
        auto Db = HuffConfig::Db_deflate;
        lossless::wrapper::encode_deflate<Huff>
            <<<get_Dg(num_chunks, Db), Db>>>(d__enc_space, len, d__bits_of_chunks, chunk_size);
        cudaDeviceSynchronize();
    }

    // gather metadata (without write) before gathering huff as sp on GPU
    // TODO move outside
    auto h__uninitialized = new size_t[num_chunks * 3]();

    ProcessHuffmanMetadataAfterChunkwiseEncoding<Huff>(
        h__uninitialized, d__bits_of_chunks, num_chunks, num_bits, num_units);

    // partially gather on GPU and copy back (TODO fully)
    auto h__compact_huff = new Huff[num_units]();
    {
        auto d__compact_huff = mem::create_CUDA_space<Huff>(num_units);
        auto d__units        = mem::create_devspace_memcpy_h2d(h__uninitialized, /*            */ num_chunks);
        auto d__entries      = mem::create_devspace_memcpy_h2d(h__uninitialized + num_chunks * 2, num_chunks);

        GatherEncodedChunks<<<num_chunks, 128>>>(d__enc_space, d__compact_huff, d__entries, d__units, chunk_size);
        cudaDeviceSynchronize();

        cudaMemcpy(h__compact_huff, d__compact_huff, num_units * sizeof(Huff), cudaMemcpyDeviceToHost);

        cudaFree(d__entries), cudaFree(d__units), cudaFree(d__compact_huff);
    }

    // write metadata to fs
    io::write_array_to_binary(basename + ".hmeta", h__uninitialized + num_chunks, 2 * num_chunks);
    io::write_array_to_binary(basename + ".hbyte", h__compact_huff, num_units);

    // metadata_size = (2 * num_chunks) * sizeof(decltype(h__uninitialized))             // hmeta
    //                 + sizeof(Huff) * (2 * type_bitcount) + sizeof(Quant) * book_len;  // reversebook

    // clean up
    cudaFree(d__enc_space), cudaFree(d__bits_of_chunks);
    delete[] h__compact_huff, delete[] h__uninitialized;

    // return std::make_tuple(num_bits, num_units, metadata_size);
}

}  // namespace sc21
