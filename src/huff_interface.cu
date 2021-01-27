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

#include "hist.cuh"
#include "huff_codec.cuh"
#include "huff_interface.cuh"
#include "par_huffman.cuh"
#include "type_aliasing.hh"
#include "type_trait.hh"
#include "types.hh"
#include "utils/cuda_err.cuh"
#include "utils/cuda_mem.cuh"
#include "utils/dbg_print.cuh"
#include "utils/format.hh"
#include "utils/io.hh"
#include "utils/timer.hh"

#include "cascaded.hpp"
#include "nvcomp.hpp"

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

typedef std::tuple<size_t, size_t, size_t, bool> tuple_3ul_1bool;
namespace kernel = data_process::reduce;

template <typename Input>
void lossless::wrapper::GetFrequency(Input* d_in, size_t len, unsigned int* d_freq, int dict_size)
{
    static_assert(
        std::is_same<Input, UI1>::value         //
            or std::is_same<Input, UI2>::value  //
            or std::is_same<Input, I1>::value   //
            or std::is_same<Input, I2>::value,
        "To get frequency, input dtype must be uint/int{8,16}_t");

    // Parameters for thread and block count optimization
    // Initialize to device-specific values
    int deviceId, max_bytes, max_bytes_opt_in, num_SMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&max_bytes, cudaDevAttrMaxSharedMemoryPerBlock, deviceId);
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, deviceId);

    // Account for opt-in extra shared memory on certain architectures
    cudaDeviceGetAttribute(&max_bytes_opt_in, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);
    max_bytes = std::max(max_bytes, max_bytes_opt_in);

    // Optimize launch
    int num_buckets      = dict_size;
    int num_values       = len;
    int items_per_thread = 1;
    int r_per_block      = (max_bytes / (int)sizeof(int)) / (num_buckets + 1);
    int num_blocks       = num_SMs;
    // fits to size
    int threads_per_block = ((((num_values / (num_blocks * items_per_thread)) + 1) / 64) + 1) * 64;
    while (threads_per_block > 1024) {
        if (r_per_block <= 1) { threads_per_block = 1024; }
        else {
            r_per_block /= 2;
            num_blocks *= 2;
            threads_per_block = ((((num_values / (num_blocks * items_per_thread)) + 1) / 64) + 1) * 64;
        }
    }

    if CONSTEXPR (
        std::is_same<Input, UI1>::value     //
        or std::is_same<Input, UI2>::value  //
        or std::is_same<Input, UI4>::value) {
        cudaFuncSetAttribute(
            kernel::p2013Histogram<Input, unsigned int>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_bytes);
        kernel::p2013Histogram                                                                    //
            <<<num_blocks, threads_per_block, ((num_buckets + 1) * r_per_block) * sizeof(int)>>>  //
            (d_in, d_freq, num_values, num_buckets, r_per_block);
    }
    else if CONSTEXPR (
        std::is_same<Input, I1>::value     //
        or std::is_same<Input, I2>::value  //
        or std::is_same<Input, I4>::value) {
        cudaFuncSetAttribute(
            kernel::p2013Histogram_int_input<Input, unsigned int>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            max_bytes);
        kernel::p2013Histogram_int_input                                                          //
            <<<num_blocks, threads_per_block, ((num_buckets + 1) * r_per_block) * sizeof(int)>>>  //
            (d_in, d_freq, num_values, num_buckets, r_per_block, dict_size / 2);
    }
    else {
        LogAll(log_err, "must be Signed or Unsigned integer as Input type");
    }

    cudaDeviceSynchronize();

#ifdef DEBUG_PRINT
    print_histogram<unsigned int><<<1, 32>>>(d_freq, dict_size, dict_size / 2);
    cudaDeviceSynchronize();
#endif
}

template <typename Huff>
void lossless::utils::PrintChunkHuffmanCoding(
    size_t* dH_bit_meta,  //
    size_t* dH_uInt_meta,
    size_t  len,
    int     chunk_size,
    size_t  total_bits,
    size_t  total_uInts)
{
    cout << "\n" << log_dbg << "Huffman coding detail start ------" << endl;
    printf("| %s\t%s\t%s\t%s\t%9s\n", "chunk", "bits", "bytes", "uInt", "chunkCR");
    for (size_t i = 0; i < 8; i++) {
        size_t n_byte   = (dH_bit_meta[i] - 1) / 8 + 1;
        auto   chunk_CR = ((double)chunk_size * sizeof(float) / (1.0 * (double)dH_uInt_meta[i] * sizeof(Huff)));
        printf("| %lu\t%lu\t%lu\t%lu\t%9.6lf\n", i, dH_bit_meta[i], n_byte, dH_uInt_meta[i], chunk_CR);
    }
    cout << "| ..." << endl
         << "| Huff.total.bits:\t" << total_bits << endl
         << "| Huff.total.bytes:\t" << total_uInts * sizeof(Huff) << endl
         << "| Huff.CR (uInt):\t" << (double)len * sizeof(float) / (total_uInts * 1.0 * sizeof(Huff)) << endl;
    cout << log_dbg << "coding detail end ----------------" << endl;
    cout << endl;
}

template <typename Quant, typename Huff, typename Data>
tuple_3ul_1bool lossless::interface::HuffmanEncode(
    string& basename,
    Quant*  d_in,
    size_t  len,
    int     chunk_size,
    bool    to_nvcomp,
    int     dict_size,
    bool    export_cb)
{
    // histogram
    auto d_freq = mem::CreateCUDASpace<unsigned int>(dict_size);
    lossless::wrapper::GetFrequency(d_in, len, d_freq, dict_size);

    auto d_canon_cb = mem::CreateCUDASpace<Huff>(dict_size, 0xff);
    // canonical Huffman; follows H to decide first and entry type
    auto type_bitcount = sizeof(Huff) * 8;
    // first, entry, reversed codebook; CHANGED first and entry to H type
    auto nbyte_dec_ancillary = sizeof(Huff) * (2 * type_bitcount) + sizeof(Quant) * dict_size;
    auto d_dec_ancillary     = mem::CreateCUDASpace<uint8_t>(nbyte_dec_ancillary);

    // Get codebooks
    lossless::par_huffman::ParGetCodebook<Quant, Huff>(dict_size, d_freq, d_canon_cb, d_dec_ancillary);
    cudaDeviceSynchronize();

    auto decode_meta = mem::CreateHostSpaceAndMemcpyFromDevice(d_dec_ancillary, nbyte_dec_ancillary);

    // Non-deflated output
    auto d_huff_space = mem::CreateCUDASpace<Huff>(len);

    if (export_cb) {  // internal evaluation, not stored in sz archive
        auto              cb_dump = mem::CreateHostSpaceAndMemcpyFromDevice(d_canon_cb, dict_size);
        std::stringstream s;
        s << basename + "-" << dict_size << "-ui" << sizeof(Huff) << ".lean_cb";
        LogAll(log_dbg, "export \"lean\" codebook (of dict_size) as", s.str());
        io::WriteArrayToBinary(s.str(), cb_dump, dict_size);
        delete[] cb_dump;
        cb_dump = nullptr;
    }

    // fix-length space
    {
        auto Db = HuffConfig::Db_encode;
        auto Dg = (len - 1) / Db + 1;
        if CONSTEXPR (std::is_same<Quant, UI1>::value or std::is_same<Quant, UI2>::value) {
            lossless::wrapper::EncodeFixedLen<Quant, Huff><<<Dg, Db>>>(d_in, d_huff_space, len, d_canon_cb);
        }
        else if CONSTEXPR (std::is_same<Quant, I1>::value or std::is_same<Quant, I2>::value) {
            lossless::wrapper::EncodeFixedLen<Quant, Huff>
                <<<Dg, Db>>>(d_in, d_huff_space, len, d_canon_cb, dict_size / 2);
        }
        cudaDeviceSynchronize();
    }

    // deflate
    auto nchunk           = (len - 1) / chunk_size + 1;  // |
    auto d_huff_bitcounts = mem::CreateCUDASpace<size_t>(nchunk);
    {
        auto Db = HuffConfig::Db_deflate;
        auto Dg = (nchunk - 1) / Db + 1;
        lossless::wrapper::Deflate<Huff><<<Dg, Db>>>(d_huff_space, len, d_huff_bitcounts, chunk_size);
        cudaDeviceSynchronize();
    }

    // TODO gather on GPU

    // dump TODO change to int
    auto _counts          = new size_t[nchunk * 3]();
    auto huff_uint_counts = _counts, huff_bitcounts = _counts + nchunk, huff_uint_entries = _counts + nchunk * 2;

    cudaMemcpy(huff_bitcounts, d_huff_bitcounts, nchunk * sizeof(size_t), cudaMemcpyDeviceToHost);
    memcpy(huff_uint_counts, huff_bitcounts, nchunk * sizeof(size_t));
    for_each(huff_uint_counts, huff_uint_counts + nchunk, [&](size_t& i) { i = (i - 1) / (sizeof(Huff) * 8) + 1; });
    // inclusive scan
    memcpy(huff_uint_entries + 1, huff_uint_counts, (nchunk - 1) * sizeof(size_t));
    for (auto i = 1; i < nchunk; i++) huff_uint_entries[i] += huff_uint_entries[i - 1];

    // sum bits from each chunk
    auto total_bits  = std::accumulate(huff_bitcounts, huff_bitcounts + nchunk, (size_t)0);
    auto total_uInts = std::accumulate(huff_uint_counts, huff_uint_counts + nchunk, (size_t)0);

    auto fmt_enc1 = "Huffman enc: (#) " + std::to_string(nchunk) + " x " + std::to_string(chunk_size);
    auto fmt_enc2 = std::to_string(total_uInts) + " " + std::to_string(sizeof(Huff)) + "-byte words or " +
                    std::to_string(total_bits) + " bits";
    LogAll(log_dbg, fmt_enc1, "=>", fmt_enc2);

    // print densely metadata
    // PrintChunkHuffmanCoding<H>(huff_bitcounts, huff_uint_counts, len, chunk_size, total_bits, total_uInts);

    // copy back densely Huffman code in units of uInt (regarding endianness)
    // TODO reinterpret_cast
    auto h = new Huff[total_uInts]();
    for (auto i = 0; i < nchunk; i++) {
        cudaMemcpy(
            h + huff_uint_entries[i],            // dst
            d_huff_space + i * chunk_size,       // src
            huff_uint_counts[i] * sizeof(Huff),  // len in H-uint
            cudaMemcpyDeviceToHost);
    }

    bool nvcomp_in_use = false;
    if (to_nvcomp) {
        int*         uncompressed_data;
        const size_t in_bytes = sizeof(Huff) * total_uInts;
        cudaMalloc(&uncompressed_data, in_bytes);
        cudaMemcpy(uncompressed_data, h, in_bytes, cudaMemcpyHostToDevice);
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

        delete[] h;
        total_uInts = output_size / sizeof(Huff);
        h           = new Huff[total_uInts]();
        cudaMemcpy(h, output_space, output_size, cudaMemcpyDeviceToHost);
        cudaFree(uncompressed_data);
        cudaFree(temp_space);
        cudaFree(output_space);
        cudaStreamDestroy(stream);

        // record nvcomp status in metadata
        // TODO nvcomp_in_use is to export: rename it
        nvcomp_in_use = true;
    }

    auto time_a = hires::now();
    // dump bit_meta and uInt_meta
    io::WriteArrayToBinary(basename + ".hmeta", _counts + nchunk, (2 * nchunk));
    // write densely Huffman code and its metadata
    io::WriteArrayToBinary(basename + ".hbyte", h, total_uInts);
    // to save first, entry and keys
    io::WriteArrayToBinary(
        basename + ".canon",                                            //
        reinterpret_cast<uint8_t*>(decode_meta),                        //
        sizeof(Huff) * (2 * type_bitcount) + sizeof(Quant) * dict_size  // first, entry, reversed dict (keys)
    );
    auto time_z = hires::now();
    LogAll(log_dbg, "time writing Huff. binary:", static_cast<duration_t>(time_z - time_a).count(), "sec");

    size_t metadata_size = (2 * nchunk) * sizeof(decltype(_counts))                           //
                           + sizeof(Huff) * (2 * type_bitcount) + sizeof(Quant) * dict_size;  // uint8_t

    //////// clean up
    cudaFree(d_freq);
    cudaFree(d_canon_cb);
    cudaFree(d_dec_ancillary);
    cudaFree(d_huff_space);
    cudaFree(d_huff_bitcounts);
    delete[] h;
    delete[] _counts;
    delete[] decode_meta;

    return std::make_tuple(total_bits, total_uInts, metadata_size, nvcomp_in_use);
}

/**
 * @brief experiment warpup; use after dual-quant; of anysize
 * @todo experiment only, no decoding yet
 */
template <typename Quant, typename Huff, typename Data>
void lossless::interface::HuffmanEncodeWithTree_3D(
    Index<3>::idx_t idx,
    string&         basename,
    Quant*          h_q_in,
    size_t          len,
    int             dict_size)
{
    auto d_quant_in = mem::CreateDeviceSpaceAndMemcpyFromHost(h_q_in, len);

    auto d_freq = mem::CreateCUDASpace<unsigned int>(dict_size);
    lossless::wrapper::GetFrequency(d_quant_in, len, d_freq, dict_size);
    auto h_freq = mem::CreateHostSpaceAndMemcpyFromDevice(d_freq, dict_size);

    auto entropy = GetEntropyFromFrequency(h_freq, len, dict_size);

    std::stringstream s;
    s << basename + "-" << dict_size << "-ui" << sizeof(Huff) << ".lean_cb";
    auto h_cb = io::ReadBinaryToNewArray<Huff>(s.str(), dict_size);

    auto GetBitcount = [&](Quant& q) { return (size_t) * ((uint8_t*)&h_cb[q] + sizeof(Huff) - 1); };

    double total_bitcounts = 0;
    for (auto i = 0; i < len; i++) { total_bitcounts += GetBitcount(h_q_in[i]); }
    auto nbytes   = total_bitcounts / 8;
    auto cr_quant = len * sizeof(Quant) / nbytes;
    auto cr_data  = len * sizeof(Data) / nbytes;
    LogAll(
        log_exp,                                                //
        idx._0, idx._1, idx._2, "\t",                           //
        std::setprecision(4),                                   //
        " entropy:", entropy,                                   //
        " \e[1mavg bitcount:", total_bitcounts / len, "\e[0m",  //
        " total bitcount:", total_bitcounts,                    //
        " nbytes:", nbytes,                                     //
        " CR against quant and data:", cr_quant, cr_data);

    // nvcomp start
    //    int*         uncompressed_data;
    //    const size_t in_bytes = sizeof(Huff) * total_uInts;
    //    cudaMalloc(&uncompressed_data, in_bytes);
    //    cudaMemcpy(uncompressed_data, h, in_bytes, cudaMemcpyHostToDevice);
    //    cudaStream_t stream;
    //    cudaStreamCreate(&stream);
    //    // 2 layers RLE, 1 Delta encoding, bitpacking enabled
    //    nvcomp::CascadedCompressor<int> compressor(uncompressed_data, in_bytes / sizeof(int), 2, 1, true);
    //    const size_t                    temp_size = compressor.get_temp_size();
    //    void*                           temp_space;
    //    cudaMalloc(&temp_space, temp_size);
    //    size_t output_size = compressor.get_max_output_size(temp_space, temp_size);
    //    void*  output_space;
    //    cudaMalloc(&output_space, output_size);
    //    compressor.compress_async(temp_space, temp_size, output_space, &output_size, stream);
    //    cudaStreamSynchronize(stream);
    //
    //    delete[] h;
    //    total_uInts = output_size / sizeof(Huff);
    //    h           = new Huff[total_uInts]();
    //    cudaMemcpy(h, output_space, output_size, cudaMemcpyDeviceToHost);
    //    cudaFree(uncompressed_data);
    //    cudaFree(temp_space);
    //    cudaFree(output_space);
    //    cudaStreamDestroy(stream);

    // nvcomp end

    cudaFree(d_freq);
    cudaFree(d_quant_in);
}

template <typename Quant, typename Huff, typename Data>
Quant* lossless::interface::HuffmanDecode(
    std::string& basename,  //
    size_t       len,
    int          chunk_size,
    int          total_uInts,
    bool         nvcomp_in_use,
    int          dict_size)
{
    auto type_bw        = sizeof(Huff) * 8;
    auto canonical_meta = sizeof(Huff) * (2 * type_bw) + sizeof(Quant) * dict_size;
    auto canonical_byte = io::ReadBinaryToNewArray<uint8_t>(basename + ".canon", canonical_meta);
    cudaDeviceSynchronize();

    auto n_chunk         = (len - 1) / chunk_size + 1;
    auto huff_multibyte  = io::ReadBinaryToNewArray<Huff>(basename + ".hbyte", total_uInts);
    auto huff_chunk_meta = io::ReadBinaryToNewArray<size_t>(basename + ".hmeta", 2 * n_chunk);
    auto Db              = HuffConfig::Db_deflate;  // the same as deflating
    auto Dg              = (n_chunk - 1) / Db + 1;

    auto d_xq             = mem::CreateCUDASpace<Quant>(len);
    auto d_huff_multibyte = mem::CreateDeviceSpaceAndMemcpyFromHost(huff_multibyte, total_uInts);

    // if nvcomp is used to compress *.hbyte
    if (nvcomp_in_use) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        nvcomp::Decompressor<int> decompressor(d_huff_multibyte, total_uInts * sizeof(Huff), stream);
        const size_t              temp_size = decompressor.get_temp_size();
        void*                     temp_space;
        cudaMalloc(&temp_space, temp_size);

        const size_t output_count = decompressor.get_num_elements();
        int*         output_space;
        cudaMalloc((void**)&output_space, output_count * sizeof(int));

        decompressor.decompress_async(temp_space, temp_size, output_space, output_count, stream);

        cudaStreamSynchronize(stream);
        cudaFree(d_huff_multibyte);

        d_huff_multibyte = mem::CreateCUDASpace<Huff>((unsigned long)(output_count * sizeof(int)));
        cudaMemcpy(d_huff_multibyte, output_space, output_count * sizeof(int), cudaMemcpyDeviceToDevice);
        total_uInts = output_count * sizeof(int) / sizeof(Huff);

        cudaFree(output_space);

        cudaStreamDestroy(stream);
        cudaFree(temp_space);
    }

    auto d_huff_chunk_meta = mem::CreateDeviceSpaceAndMemcpyFromHost(huff_chunk_meta, 2 * n_chunk);
    auto d_canonical_byte  = mem::CreateDeviceSpaceAndMemcpyFromHost(canonical_byte, canonical_meta);
    cudaDeviceSynchronize();

    lossless::wrapper::Decode<<<Dg, Db, canonical_meta>>>(  //
        d_huff_multibyte, d_huff_chunk_meta, d_xq, len, chunk_size, n_chunk, d_canonical_byte, (size_t)canonical_meta);
    cudaDeviceSynchronize();

    auto xq = mem::CreateHostSpaceAndMemcpyFromDevice(d_xq, len);
    cudaFree(d_xq);
    cudaFree(d_huff_multibyte);
    cudaFree(d_huff_chunk_meta);
    cudaFree(d_canonical_byte);
    delete[] huff_multibyte;
    delete[] huff_chunk_meta;
    delete[] canonical_byte;

    return xq;
}

// TODO mark types using Q/H-byte binding; internally resolve UI8-UI8_2 issue
// using Q1 = QuantTrait<1>::Quant;
// using H4 = HuffTrait<4>::Huff;

// clang-format off
template tuple_3ul_1bool lossless::interface::HuffmanEncode<UI1, UI4, FP4>(string&, UI1*, size_t, int, bool, int, bool);
template tuple_3ul_1bool lossless::interface::HuffmanEncode<UI2, UI4, FP4>(string&, UI2*, size_t, int, bool, int, bool);
template tuple_3ul_1bool lossless::interface::HuffmanEncode<UI1, UI8, FP4>(string&, UI1*, size_t, int, bool, int, bool);
template tuple_3ul_1bool lossless::interface::HuffmanEncode<UI2, UI8, FP4>(string&, UI2*, size_t, int, bool, int, bool);

template UI1* lossless::interface::HuffmanDecode<UI1, UI4, FP4>(std::string&, size_t, int, int, bool, int);
template UI2* lossless::interface::HuffmanDecode<UI2, UI4, FP4>(std::string&, size_t, int, int, bool, int);
template UI1* lossless::interface::HuffmanDecode<UI1, UI8, FP4>(std::string&, size_t, int, int, bool, int);
template UI2* lossless::interface::HuffmanDecode<UI2, UI8, FP4>(std::string&, size_t, int, int, bool, int);

template void lossless::interface::HuffmanEncodeWithTree_3D<UI1, UI4>(Index<3>::idx_t, string&, UI1*, size_t, int);
template void lossless::interface::HuffmanEncodeWithTree_3D<UI1, UI8>(Index<3>::idx_t, string&, UI1*, size_t, int);
template void lossless::interface::HuffmanEncodeWithTree_3D<UI2, UI4>(Index<3>::idx_t, string&, UI2*, size_t, int);
template void lossless::interface::HuffmanEncodeWithTree_3D<UI2, UI8>(Index<3>::idx_t, string&, UI2*, size_t, int);
