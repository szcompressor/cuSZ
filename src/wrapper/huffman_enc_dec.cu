/**
 * @file huffman_enc_dec.cu
 * @author Jiannan Tian, Cody Rivera (cjrivera1@crimson.ua.edu)
 * @brief Workflow of Huffman coding.
 * @version 0.1
 * @date 2020-10-24
 * (created) 2020-04-24 (rev) 2021-09-0
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cuda_runtime.h>

#include <sys/stat.h>
#include <unistd.h>
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

#include "../kernel/hist.h"
#include "../kernel/huffman_codec.h"
#include "../type_aliasing.hh"
#include "../type_trait.hh"
#include "../types.hh"
#include "../utils.hh"
#include "huffman_enc_dec.cuh"

#ifdef MODULAR_ELSEWHERE
#include "cascaded.hpp"
#include "nvcomp.hpp"
#endif

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

#define nworker blockDim.x

template <typename Huff>
__global__ void cusz::huffman_enc_concatenate(
    Huff*   in_enc_space,
    Huff*   out_bitstream,
    size_t* sp_entries,
    size_t* sp_uints,
    size_t  chunk_size)
{
    auto len      = sp_uints[blockIdx.x];
    auto sp_entry = sp_entries[blockIdx.x];
    auto dn_entry = chunk_size * blockIdx.x;

    for (auto i = 0; i < (len + nworker - 1) / nworker; i++) {
        auto _tid = threadIdx.x + i * nworker;
        if (_tid < len) *(out_bitstream + sp_entry + _tid) = *(in_enc_space + dn_entry + _tid);
        __syncthreads();
    }
}

template <typename Huff>
void cusz::huffman_process_metadata(
    size_t* _counts,
    size_t* dev_bits,
    size_t  nchunk,
    size_t& num_bits,
    size_t& num_uints)
{
    constexpr auto TYPE_BITCOUNT = sizeof(Huff) * 8;

    auto sp_uints = _counts, sp_bits = _counts + nchunk, sp_entries = _counts + nchunk * 2;

    cudaMemcpy(sp_bits, dev_bits, nchunk * sizeof(size_t), cudaMemcpyDeviceToHost);
    memcpy(sp_uints, sp_bits, nchunk * sizeof(size_t));
    for_each(sp_uints, sp_uints + nchunk, [&](size_t& i) { i = (i + TYPE_BITCOUNT - 1) / TYPE_BITCOUNT; });
    memcpy(sp_entries + 1, sp_uints, (nchunk - 1) * sizeof(size_t));
    for (auto i = 1; i < nchunk; i++) sp_entries[i] += sp_entries[i - 1];  // inclusive scan

    num_bits  = std::accumulate(sp_bits, sp_bits + nchunk, (size_t)0);
    num_uints = std::accumulate(sp_uints, sp_uints + nchunk, (size_t)0);
}

/*
template <typename T>
void draft::UseNvcompZip(T* space, size_t& len)
{
    int*         uncompressed_data;
    const size_t in_bytes = len * sizeof(T);

    cudaMalloc(&uncompressed_data, in_bytes);
    cudaMemcpy(uncompressed_data, space, in_bytes, cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // 2 layers RLE, 1 Delta encoding, bitpacking enabled
    nvcomp::CascadedCompressor<int> compressor(uncompressed_data, in_bytes / sizeof(int), 2, 1, true);
    const size_t                    temp_size = compressor.get_temp_size();
    void*                           temp_space;
    cudaMalloc(&temp_space, temp_size);
    size_t output_size = compressor.get_max_output_size(temp_space, temp_size);
    void*  output_space;
    cudaMalloc(&output_space, output_size);
    compressor.compress_async(temp_space, temp_size, output_space, &output_size, stream);
    cudaStreamSynchronize(stream);
    // TODO ad hoc; should use original GPU space
    memset(space, 0x0, len * sizeof(T));
    len = output_size / sizeof(T);
    cudaMemcpy(space, output_space, output_size, cudaMemcpyDeviceToHost);

    cudaFree(uncompressed_data);
    cudaFree(temp_space);
    cudaFree(output_space);
    cudaStreamDestroy(stream);
}

template <typename T>
void draft::UseNvcompUnzip(T** d_space, size_t& len)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    nvcomp::Decompressor<int> decompressor(*d_space, len * sizeof(T), stream);
    const size_t              temp_size = decompressor.get_temp_size();
    void*                     temp_space;
    cudaMalloc(&temp_space, temp_size);

    const size_t output_count = decompressor.get_num_elements();
    int*         output_space;
    cudaMalloc((void**)&output_space, output_count * sizeof(int));

    decompressor.decompress_async(temp_space, temp_size, output_space, output_count, stream);

    cudaStreamSynchronize(stream);
    cudaFree(*d_space);

    *d_space = mem::create_CUDA_space<T>((unsigned long)(output_count * sizeof(int)));
    cudaMemcpy(*d_space, output_space, output_count * sizeof(int), cudaMemcpyDeviceToDevice);
    len = output_count * sizeof(int) / sizeof(T);

    cudaFree(output_space);

    cudaStreamDestroy(stream);
    cudaFree(temp_space);
}

*/

template <typename Quant, typename Huff, typename Data>
void lossless::HuffmanEncode(
    string& basename,
    Quant*  dev_input,
    Huff*   dev_book,
    size_t  len,
    int     chunk_size,
    int     dict_size,
    size_t& out_num_bits,
    size_t& out_num_uints,
    size_t& out_metadata_size,
    float&  milliseconds)
{
    constexpr auto TYPE_BITCOUNT = sizeof(Huff) * 8;  // canonical Huffman; follows H to decide first and entry type

    auto get_grid_dim = [](size_t problem_size, size_t block_dim) {
        return (problem_size + block_dim - 1) / block_dim;
    };

    // Huffman space in dense format (full of zeros), fix-length space
    auto dev_enc_space =
        mem::create_CUDA_space<Huff>(len + chunk_size + HuffConfig::Db_encode);  // TODO ad hoc (big) padding
    {
        auto block_dim = HuffConfig::Db_encode;
        auto t         = new cuda_timer_t;
        t->timer_start();
        cusz::encode_fixedlen_space_cub<Quant, Huff, HuffConfig::enc_sequentiality>
            <<<get_grid_dim(len, block_dim), block_dim / HuffConfig::enc_sequentiality>>>(
                dev_input, dev_enc_space, len, dev_book);

        milliseconds += t->timer_end_get_elapsed_time();
        cudaDeviceSynchronize();
        delete t;
    }

    // encode_deflate
    auto nchunk   = (len + chunk_size - 1) / chunk_size;
    auto dev_bits = mem::create_CUDA_space<size_t>(nchunk);
    {
        auto block_dim = HuffConfig::Db_deflate;
        auto t         = new cuda_timer_t;
        t->timer_start();
        cusz::encode_deflate<Huff>
            <<<get_grid_dim(nchunk, block_dim), block_dim>>>(dev_enc_space, len, dev_bits, chunk_size);
        milliseconds += t->timer_end_get_elapsed_time();
        cudaDeviceSynchronize();
        delete t;
    }

    // gather metadata (without write) before gathering huff as sp on GPU
    auto   _counts  = new size_t[nchunk * 3]();
    size_t num_bits = 0, num_uints = 0;
    cusz::huffman_process_metadata<Huff>(_counts, dev_bits, nchunk, num_bits, num_uints);

    // partially gather on GPU and copy back (TODO fully)
    auto host_out_bitstream = new Huff[num_uints]();
    {
        auto dev_out_bitstream = mem::create_CUDA_space<Huff>(num_uints);
        auto dev_uints         = mem::create_devspace_memcpy_h2d(_counts, nchunk);               // sp_uints
        auto dev_entries       = mem::create_devspace_memcpy_h2d(_counts + nchunk * 2, nchunk);  // sp_entries
        cusz::huffman_enc_concatenate<<<nchunk, 128>>>(
            dev_enc_space, dev_out_bitstream, dev_entries, dev_uints, chunk_size);
        cudaDeviceSynchronize();
        cudaMemcpy(host_out_bitstream, dev_out_bitstream, num_uints * sizeof(Huff), cudaMemcpyDeviceToHost);
        cudaFree(dev_entries), cudaFree(dev_uints), cudaFree(dev_out_bitstream);
    }

    // write metadata to fs
    io::write_array_to_binary(basename + ".hmeta", _counts + nchunk, 2 * nchunk);
    io::write_array_to_binary(basename + ".hbyte", host_out_bitstream, num_uints);

    size_t metadata_size =
        (2 * nchunk) * sizeof(decltype(_counts)) + sizeof(Huff) * (2 * TYPE_BITCOUNT) + sizeof(Quant) * dict_size;

    // clean up
    cudaFree(dev_enc_space), cudaFree(dev_bits);
    delete[] host_out_bitstream, delete[] _counts;

    out_num_bits      = num_bits;
    out_num_uints     = num_uints;
    out_metadata_size = metadata_size;
}

template <typename Quant, typename Huff, typename Data>
void lossless::HuffmanDecode(
    std::string&               basename,  //
    struct PartialData<Quant>* quant,
    size_t                     len,
    int                        chunk_size,
    size_t                     num_uints,
    int                        dict_size,
    float&                     milliseconds)
{
    constexpr auto TYPE_BITCOUNT = sizeof(Huff) * 8;
    auto           revbook_nbyte = sizeof(Huff) * (2 * TYPE_BITCOUNT) + sizeof(Quant) * dict_size;
    auto           host_revbook  = io::read_binary_to_new_array<uint8_t>(basename + ".canon", revbook_nbyte);

    auto nchunk            = (len - 1) / chunk_size + 1;
    auto host_in_bitstream = io::read_binary_to_new_array<Huff>(basename + ".hbyte", num_uints);
    auto host_bits_entries = io::read_binary_to_new_array<size_t>(basename + ".hmeta", 2 * nchunk);
    auto block_dim         = HuffConfig::Db_deflate;  // the same as deflating
    auto grid_dim          = (nchunk - 1) / block_dim + 1;

    auto dev_out_bitstream = mem::create_devspace_memcpy_h2d(host_in_bitstream, num_uints);
    auto dev_bits_entries  = mem::create_devspace_memcpy_h2d(host_bits_entries, 2 * nchunk);
    auto dev_revbook       = mem::create_devspace_memcpy_h2d(host_revbook, revbook_nbyte);
    cudaDeviceSynchronize();

    {
        auto t = new cuda_timer_t;
        t->timer_start();
        cusz::Decode<<<grid_dim, block_dim, revbook_nbyte>>>(  //
            dev_out_bitstream, dev_bits_entries, quant->dptr, len, chunk_size, nchunk, dev_revbook,
            (size_t)revbook_nbyte);
        milliseconds += t->timer_end_get_elapsed_time();
        cudaDeviceSynchronize();
        delete t;
    }

    cudaFree(dev_out_bitstream);
    cudaFree(dev_bits_entries);
    cudaFree(dev_revbook);
    delete[] host_in_bitstream;
    delete[] host_bits_entries;
    delete[] host_revbook;
}

// TODO mark types using Q/H-byte binding; internally resolve UI8-UI8_2 issue

#define HUFFMAN_ENCODE(Q, H, D)                     \
    template void lossless::HuffmanEncode<Q, H, D>( \
        string&, Q*, H*, size_t, int, int, size_t&, size_t&, size_t&, float&);

HUFFMAN_ENCODE(UI1, UI4, FP4)
HUFFMAN_ENCODE(UI2, UI4, FP4)
HUFFMAN_ENCODE(UI1, UI8, FP4)
HUFFMAN_ENCODE(UI2, UI8, FP4)

#define HUFFMAN_DECODE(Q, H, D)                     \
    template void lossless::HuffmanDecode<Q, H, D>( \
        std::string&, struct PartialData<Q>*, size_t, int, size_t, int, float&);

HUFFMAN_DECODE(UI1, UI4, FP4)
HUFFMAN_DECODE(UI2, UI4, FP4)
HUFFMAN_DECODE(UI1, UI8, FP4)
HUFFMAN_DECODE(UI2, UI8, FP4)
