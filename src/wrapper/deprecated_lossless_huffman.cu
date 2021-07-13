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
#include "deprecated_lossless_huffman.h"

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
__global__ void cusz::CopyHuffmanUintsDenseToSparse(
    Huff*   input_dn,
    Huff*   output_sp,
    size_t* sp_entries,
    size_t* sp_uints,
    size_t  dn_chunk_size)
{
    auto len      = sp_uints[blockIdx.x];
    auto sp_entry = sp_entries[blockIdx.x];
    auto dn_entry = dn_chunk_size * blockIdx.x;

    for (auto i = 0; i < (len + nworker - 1) / nworker; i++) {
        auto _tid = threadIdx.x + i * nworker;
        if (_tid < len) *(output_sp + sp_entry + _tid) = *(input_dn + dn_entry + _tid);
        __syncthreads();
    }
}

template <typename Huff>
void cusz::GatherSpHuffMetadata(
    size_t* _counts,
    size_t* d_sp_bits,
    size_t  nchunk,
    size_t& total_bits,
    size_t& total_uints)
{
    static const size_t Huff_bytes = sizeof(Huff) * 8;

    auto sp_uints = _counts, sp_bits = _counts + nchunk, sp_entries = _counts + nchunk * 2;

    cudaMemcpy(sp_bits, d_sp_bits, nchunk * sizeof(size_t), cudaMemcpyDeviceToHost);
    memcpy(sp_uints, sp_bits, nchunk * sizeof(size_t));
    for_each(sp_uints, sp_uints + nchunk, [&](size_t& i) { i = (i + Huff_bytes - 1) / Huff_bytes; });
    memcpy(sp_entries + 1, sp_uints, (nchunk - 1) * sizeof(size_t));
    for (auto i = 1; i < nchunk; i++) sp_entries[i] += sp_entries[i - 1];  // inclusive scan

    total_bits  = std::accumulate(sp_bits, sp_bits + nchunk, (size_t)0);
    total_uints = std::accumulate(sp_uints, sp_uints + nchunk, (size_t)0);

    //    auto fmt_enc1 = "Huffman enc: (#) " + std::to_string(nchunk) + " x " + std::to_string(dn_chunk);
    //    auto fmt_enc2 = std::to_string(total_uints) + " " + std::to_string(sizeof(Huff)) + "-byte words or " +
    //                    std::to_string(total_bits) + " bits";
    //    logging(log_dbg, fmt_enc1, "=>", fmt_enc2);
}

#ifdef MODULAR_ELSEWHERE
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

#endif

template <typename Quant, typename Huff, typename Data>
std::tuple<size_t, size_t, size_t> lossless::interface::HuffmanEncode(
    string&  basename,
    Quant*   d_input,
    Huff*    d_canon_cb,
    uint8_t* d_reverse_cb,
    size_t   _nbyte,
    size_t   len,
    int      dn_chunk,
    int      dict_size,
    float&   milliseconds)
{
    static const auto type_bitcount = sizeof(Huff) * 8;  // canonical Huffman; follows H to decide first and entry type

    auto get_Dg = [](size_t problem_size, size_t Db) { return (problem_size + Db - 1) / Db; };

    auto decode_meta = mem::create_devspace_memcpy_d2h(d_reverse_cb, _nbyte);
    io::write_array_to_binary(
        basename + ".canon", reinterpret_cast<uint8_t*>(decode_meta),
        sizeof(Huff) * (2 * type_bitcount) + sizeof(Quant) * dict_size);
    delete[] decode_meta;

    // Huffman space in dense format (full of zeros), fix-length space
    auto d_huff_dn = mem::create_CUDA_space<Huff>(len + dn_chunk + HuffConfig::Db_encode);  // TODO ad hoc (big) padding
    {
        auto Db = HuffConfig::Db_encode;

        auto huffenc_fixlen = new cuda_timer_t;
        huffenc_fixlen->timer_start();

        cusz::encode_fixedlen_space_cub<Quant, Huff, HuffConfig::enc_sequentiality>
            <<<get_Dg(len, Db), Db / HuffConfig::enc_sequentiality>>>(d_input, d_huff_dn, len, d_canon_cb);

        milliseconds += huffenc_fixlen->timer_end_get_elapsed_time();

        cudaDeviceSynchronize();

        delete huffenc_fixlen;
    }

    // encode_deflate
    auto nchunk    = (len + dn_chunk - 1) / dn_chunk;
    auto d_sp_bits = mem::create_CUDA_space<size_t>(nchunk);
    {
        auto Db = HuffConfig::Db_deflate;

        auto huffenc_deflate = new cuda_timer_t;
        huffenc_deflate->timer_start();

        cusz::encode_deflate<Huff><<<get_Dg(nchunk, Db), Db>>>(d_huff_dn, len, d_sp_bits, dn_chunk);

        milliseconds += huffenc_deflate->timer_end_get_elapsed_time();

        cudaDeviceSynchronize();

        delete huffenc_deflate;
    }

    // gather metadata (without write) before gathering huff as sp on GPU
    auto   _counts    = new size_t[nchunk * 3]();
    size_t total_bits = 0, total_uints = 0;
    cusz::GatherSpHuffMetadata<Huff>(_counts, d_sp_bits, nchunk, total_bits, total_uints);

    // partially gather on GPU and copy back (TODO fully)
    auto huff_sp = new Huff[total_uints]();
    {
        auto d_huff_sp = mem::create_CUDA_space<Huff>(total_uints);
        auto d_uints   = mem::create_devspace_memcpy_h2d(_counts, nchunk);               // sp_uints
        auto d_entries = mem::create_devspace_memcpy_h2d(_counts + nchunk * 2, nchunk);  // sp_entries
        cusz::CopyHuffmanUintsDenseToSparse<<<nchunk, 128>>>(d_huff_dn, d_huff_sp, d_entries, d_uints, dn_chunk);
        cudaDeviceSynchronize();
        cudaMemcpy(huff_sp, d_huff_sp, total_uints * sizeof(Huff), cudaMemcpyDeviceToHost);
        cudaFree(d_entries), cudaFree(d_uints), cudaFree(d_huff_sp);
    }

    // write metadata to fs
    io::write_array_to_binary(basename + ".hmeta", _counts + nchunk, 2 * nchunk);
    io::write_array_to_binary(basename + ".hbyte", huff_sp, total_uints);

    size_t metadata_size =
        (2 * nchunk) * sizeof(decltype(_counts)) + sizeof(Huff) * (2 * type_bitcount) + sizeof(Quant) * dict_size;

    // clean up
    cudaFree(d_huff_dn), cudaFree(d_sp_bits);
    delete[] huff_sp, delete[] _counts;

    return std::make_tuple(total_bits, total_uints, metadata_size);
}

/**
 * @brief experiment warpup; use after dual-quant; of anysize
 * @todo experiment only, no decoding yet
 */
template <typename Quant, typename Huff, typename Data>
void lossless::interface::HuffmanEncodeWithTree_3D(
    Index<3>::idx_t idx,
    string&         basename,
    Quant*          h_quant_in,
    size_t          len,
    int             dict_size)
{
    auto d_quant_in = mem::create_devspace_memcpy_h2d(h_quant_in, len);

    auto  d_freq = mem::create_CUDA_space<unsigned int>(dict_size);
    float dummy;
    ::wrapper::get_frequency(d_quant_in, len, d_freq, dict_size, dummy);
    cudaFree(d_freq);
    auto h_freq = mem::create_devspace_memcpy_d2h(d_freq, dict_size);

    auto entropy = GetEntropyFromFrequency(h_freq, len, dict_size);

    std::stringstream s;
    s << basename + "-" << dict_size << "-ui" << sizeof(Huff) << ".lean_cb";
    auto h_cb       = io::read_binary_to_new_array<Huff>(s.str(), dict_size);
    auto d_canon_cb = mem::create_devspace_memcpy_h2d(h_cb, dict_size);

    auto get_Dg = [](size_t problem_size, size_t Db) { return (problem_size + Db - 1) / Db; };

    // Huffman space in dense format (full of zeros), fix-length space
    auto d_huff_dn = mem::create_CUDA_space<Huff>(len);
    {
        auto Db = HuffConfig::Db_encode;
        cusz::EncodeFixedLen<Quant, Huff><<<get_Dg(len, Db), Db>>>(d_quant_in, d_huff_dn, len, d_canon_cb);
        cudaDeviceSynchronize();
    }

    const static int dn_chunk = 4096;
    // encode_deflate
    auto nchunk    = (len + dn_chunk - 1) / dn_chunk;
    auto d_sp_bits = mem::create_CUDA_space<size_t>(nchunk);
    {
        auto Db = HuffConfig::Db_deflate;
        cusz::encode_deflate<Huff><<<get_Dg(nchunk, Db), Db>>>(d_huff_dn, len, d_sp_bits, dn_chunk);
        cudaDeviceSynchronize();
    }

    // gather metadata (without write) before gathering huff as sp on GPU
    auto   _counts    = new size_t[nchunk * 3]();
    size_t total_bits = 0, total_uints = 0;
    cusz::GatherSpHuffMetadata<Huff>(_counts, d_sp_bits, nchunk, total_bits, total_uints);

    // partially gather on GPU and copy back (TODO fully)
    auto huff_sp = new Huff[total_uints]();
    {
        auto d_huff_sp = mem::create_CUDA_space<Huff>(total_uints);
        auto d_uints   = mem::create_devspace_memcpy_h2d(_counts, nchunk);               // sp_uints
        auto d_entries = mem::create_devspace_memcpy_h2d(_counts + nchunk * 2, nchunk);  // sp_entries
        cusz::CopyHuffmanUintsDenseToSparse<<<nchunk, 128>>>(d_huff_dn, d_huff_sp, d_entries, d_uints, dn_chunk);
        cudaDeviceSynchronize();
        cudaMemcpy(huff_sp, d_huff_sp, total_uints * sizeof(Huff), cudaMemcpyDeviceToHost);
        cudaFree(d_entries), cudaFree(d_uints), cudaFree(d_huff_sp);
    }

    cudaFree(d_huff_dn);

    io::write_array_to_binary(
        basename + "_huff_" + std::to_string(len) + "_part_" + std::to_string(idx._0) + std::to_string(idx._1) +
            std::to_string(idx._2),
        huff_sp, total_uints);

    auto avg_bits = 1.0 * total_bits / len;
    auto cr       = 1.0 * len * sizeof(Data) / (total_uints * sizeof(Huff));

    logging(
        log_exp,                                   //
        idx._0, idx._1, idx._2, "\t",              //
        std::setprecision(4),                      //
        " \e[1mavg bitcount:", avg_bits, "\e[0m",  //
        " CR:", cr);

    delete[] huff_sp;

    cudaFree(d_freq);
    cudaFree(d_quant_in);
}

template <typename Quant, typename Huff, typename Data>
void lossless::interface::HuffmanDecode(
    std::string&               basename,  //
    struct PartialData<Quant>* quant,
    size_t                     len,
    int                        chunk_size,
    size_t                     total_uints,
    int                        dict_size,
    float&                     milliseconds)
{
    auto type_bw    = sizeof(Huff) * 8;
    auto canon_meta = sizeof(Huff) * (2 * type_bw) + sizeof(Quant) * dict_size;
    auto canon_byte = io::read_binary_to_new_array<uint8_t>(basename + ".canon", canon_meta);

    auto nchunk       = (len - 1) / chunk_size + 1;
    auto huff_sp      = io::read_binary_to_new_array<Huff>(basename + ".hbyte", total_uints);
    auto huff_sp_meta = io::read_binary_to_new_array<size_t>(basename + ".hmeta", 2 * nchunk);
    auto Db           = HuffConfig::Db_deflate;  // the same as deflating
    auto Dg           = (nchunk - 1) / Db + 1;

    auto d_huff_sp      = mem::create_devspace_memcpy_h2d(huff_sp, total_uints);
    auto d_huff_sp_meta = mem::create_devspace_memcpy_h2d(huff_sp_meta, 2 * nchunk);
    auto d_canon_byte   = mem::create_devspace_memcpy_h2d(canon_byte, canon_meta);
    cudaDeviceSynchronize();

    auto timer_huffdec = new cuda_timer_t;
    timer_huffdec->timer_start();

    cusz::Decode<<<Dg, Db, canon_meta>>>(  //
        d_huff_sp, d_huff_sp_meta, quant->dptr, len, chunk_size, nchunk, d_canon_byte, (size_t)canon_meta);

    milliseconds += timer_huffdec->timer_end_get_elapsed_time();

    cudaDeviceSynchronize();

    cudaFree(d_huff_sp);
    cudaFree(d_huff_sp_meta);
    cudaFree(d_canon_byte);
    delete[] huff_sp;
    delete[] huff_sp_meta;
    delete[] canon_byte;
}

// TODO mark types using Q/H-byte binding; internally resolve UI8-UI8_2 issue

#define HUFFMAN_ENCODE(Q, H, D)                                                              \
    template std::tuple<size_t, size_t, size_t> lossless::interface::HuffmanEncode<Q, H, D>( \
        string&, Q*, H*, uint8_t*, size_t, size_t, int, int, float&);

HUFFMAN_ENCODE(UI1, UI4, FP4)
HUFFMAN_ENCODE(UI2, UI4, FP4)
HUFFMAN_ENCODE(UI1, UI8, FP4)
HUFFMAN_ENCODE(UI2, UI8, FP4)

#define HUFFMAN_DECODE(Q, H, D)                                \
    template void lossless::interface::HuffmanDecode<Q, H, D>( \
        std::string&, struct PartialData<Q>*, size_t, int, size_t, int, float&);

HUFFMAN_DECODE(UI1, UI4, FP4)
HUFFMAN_DECODE(UI2, UI4, FP4)
HUFFMAN_DECODE(UI1, UI8, FP4)
HUFFMAN_DECODE(UI2, UI8, FP4)

// TODO remove
template void lossless::interface::HuffmanEncodeWithTree_3D<UI1, UI4>(Index<3>::idx_t, string&, UI1*, size_t, int);
template void lossless::interface::HuffmanEncodeWithTree_3D<UI1, UI8>(Index<3>::idx_t, string&, UI1*, size_t, int);
template void lossless::interface::HuffmanEncodeWithTree_3D<UI2, UI4>(Index<3>::idx_t, string&, UI2*, size_t, int);
template void lossless::interface::HuffmanEncodeWithTree_3D<UI2, UI8>(Index<3>::idx_t, string&, UI2*, size_t, int);
