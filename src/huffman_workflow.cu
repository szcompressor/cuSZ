/**
 * @file huffman_workflow.cu
 * @author Jiannan Tian, Cody Rivera (cjrivera1@crimson.ua.edu)
 * @brief Workflow of Huffman coding.
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-04-24
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cuda_runtime.h>

#include <sys/stat.h>
#include <unistd.h>
#include <bitset>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "canonical.cuh"
#include "cuda_error_handling.cuh"
#include "cuda_mem.cuh"
#include "dbg_gpu_printing.cuh"
#include "format.hh"
#include "histogram.cuh"
#include "huffman.cuh"
#include "huffman_codec.cuh"
#include "huffman_workflow.cuh"
#include "par_huffman.cuh"
#include "types.hh"

int ht_state_num;
int ht_all_nodes;
using uint8__t = uint8_t;

template <typename Q>
void wrapper::GetFrequency(Q* d_bcode, size_t len, unsigned int* d_freq, int dict_size)
{
    // Parameters for thread and block count optimization

    // Initialize to device-specific values
    int deviceId;
    int maxbytes;
    int numSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&maxbytes, cudaDevAttrMaxSharedMemoryPerMultiprocessor, deviceId);
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceId);

    // Optimize launch
    int numBuckets     = dict_size;
    int numValues      = len;
    int itemsPerThread = 1;
    int RPerBlock      = (maxbytes / (int)sizeof(int)) / (numBuckets + 1);
    int numBlocks      = numSMs;
    cudaFuncSetAttribute(p2013Histogram<Q, unsigned int>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    // fits to size
    int threadsPerBlock = ((((numValues / (numBlocks * itemsPerThread)) + 1) / 64) + 1) * 64;
    while (threadsPerBlock > 1024) {
        if (RPerBlock <= 1) {
            threadsPerBlock = 1024;
        }
        else {
            RPerBlock /= 2;
            numBlocks *= 2;
            threadsPerBlock = ((((numValues / (numBlocks * itemsPerThread)) + 1) / 64) + 1) * 64;
        }
    }
    p2013Histogram                                                                      //
        <<<numBlocks, threadsPerBlock, ((numBuckets + 1) * RPerBlock) * sizeof(int)>>>  //
        (d_bcode, d_freq, numValues, numBuckets, RPerBlock);
    cudaDeviceSynchronize();

    // TODO make entropy optional
    {
        auto   freq    = mem::CreateHostSpaceAndMemcpyFromDevice(d_freq, dict_size);
        double entropy = 0.0;
        for (auto i = 0; i < dict_size; i++)
            if (freq[i]) {
                auto possibility = freq[i] / (1.0 * len);
                entropy -= possibility * log(possibility);
            }
        cout << log_info << "entropy:\t\t" << entropy << endl;
        delete[] freq;
    }

#ifdef DEBUG_PRINT
    print_histogram<unsigned int><<<1, 32>>>(d_freq, dict_size, dict_size / 2);
    cudaDeviceSynchronize();
#endif
}

template <typename H>
void PrintChunkHuffmanCoding(
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
        auto   chunk_CR = ((double)chunk_size * sizeof(float) / (1.0 * (double)dH_uInt_meta[i] * sizeof(H)));
        printf("| %lu\t%lu\t%lu\t%lu\t%9.6lf\n", i, dH_bit_meta[i], n_byte, dH_uInt_meta[i], chunk_CR);
    }
    cout << "| ..." << endl
         << "| Huff.total.bits:\t" << total_bits << endl
         << "| Huff.total.bytes:\t" << total_uInts * sizeof(H) << endl
         << "| Huff.CR (uInt):\t" << (double)len * sizeof(float) / (total_uInts * 1.0 * sizeof(H)) << endl;
    cout << log_dbg << "coding detail end ----------------" << endl;
    cout << endl;
}

template <typename Q, typename H, typename DATA>
std::tuple<size_t, size_t, size_t> HuffmanEncode(string& f_in, Q* d_in, size_t len, int chunk_size, int dict_size)
{
    // histogram
    ht_state_num = 2 * dict_size;
    ht_all_nodes = 2 * ht_state_num;
    auto d_freq  = mem::CreateCUDASpace<unsigned int>(ht_all_nodes);
    wrapper::GetFrequency(d_in, len, d_freq, dict_size);

    // Allocate cb memory
    auto d_canonical_cb = mem::CreateCUDASpace<H>(dict_size, 0xff);
    // canonical Huffman; follows H to decide first and entry type
    auto type_bw = sizeof(H) * 8;
    // first, entry, reversed codebook
    // CHANGED first and entry to H type
    auto decode_meta_size = sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size;
    auto d_decode_meta    = mem::CreateCUDASpace<uint8_t>(decode_meta_size);

    // Get codebooks
    ParGetCodebook<Q, H>(dict_size, d_freq, d_canonical_cb, d_decode_meta);
    cudaDeviceSynchronize();

    auto decode_meta = mem::CreateHostSpaceAndMemcpyFromDevice(d_decode_meta, decode_meta_size);

    // Non-deflated output
    auto d_h = mem::CreateCUDASpace<H>(len);

    // --------------------------------
    // this is for internal evaluation, not in sz archive
    auto cb_dump = mem::CreateHostSpaceAndMemcpyFromDevice(d_canonical_cb, dict_size);
    io::WriteBinaryFile(cb_dump, dict_size, new string(f_in + ".canonized"));

    // --------------------------------

    // fix-length space
    {
        auto blockDim = tBLK_ENCODE;
        auto gridDim  = (len - 1) / blockDim + 1;
        EncodeFixedLen<Q, H><<<gridDim, blockDim>>>(d_in, d_h, len, d_canonical_cb);
        cudaDeviceSynchronize();
    }

    // deflate
    auto n_chunk       = (len - 1) / chunk_size + 1;  // |
    auto d_h_bitwidths = mem::CreateCUDASpace<size_t>(n_chunk);
    cout << log_info << "chunk.size:\t" << chunk_size << endl;
    cout << log_dbg << "chunk.num:\t" << n_chunk << endl;
    {
        auto blockDim = tBLK_DEFLATE;
        auto gridDim  = (n_chunk - 1) / blockDim + 1;
        Deflate<H><<<gridDim, blockDim>>>(d_h, len, d_h_bitwidths, chunk_size);
        cudaDeviceSynchronize();
    }

    // dump TODO change to int
    auto h_meta        = new size_t[n_chunk * 3]();
    auto dH_uInt_meta  = h_meta;
    auto dH_bit_meta   = h_meta + n_chunk;
    auto dH_uInt_entry = h_meta + n_chunk * 2;
    // copy back densely Huffman code (dHcode)
    cudaMemcpy(dH_bit_meta, d_h_bitwidths, n_chunk * sizeof(size_t), cudaMemcpyDeviceToHost);
    // transform in uInt
    memcpy(dH_uInt_meta, dH_bit_meta, n_chunk * sizeof(size_t));
    for_each(dH_uInt_meta, dH_uInt_meta + n_chunk, [&](size_t& i) { i = (i - 1) / (sizeof(H) * 8) + 1; });
    // make it entries
    memcpy(dH_uInt_entry + 1, dH_uInt_meta, (n_chunk - 1) * sizeof(size_t));
    for (auto i = 1; i < n_chunk; i++) dH_uInt_entry[i] += dH_uInt_entry[i - 1];

    // sum bits from each chunk
    auto total_bits  = std::accumulate(dH_bit_meta, dH_bit_meta + n_chunk, (size_t)0);
    auto total_uInts = std::accumulate(dH_uInt_meta, dH_uInt_meta + n_chunk, (size_t)0);

    cout << log_dbg;
    printf(
        "Huffman bitstream: %lu chunks of size = %d, in %lu uint%lus or %lu bits\n", n_chunk, chunk_size, total_uInts,
        sizeof(H) * 8, total_bits);

    // print densely metadata
    PrintChunkHuffmanCoding<H>(dH_bit_meta, dH_uInt_meta, len, chunk_size, total_bits, total_uInts);

    // copy back densely Huffman code in units of uInt (regarding endianness)
    auto h = new H[total_uInts]();
    for (auto i = 0; i < n_chunk; i++) {
        cudaMemcpy(
            h + dH_uInt_entry[i],         // dst
            d_h + i * chunk_size,         // src
            dH_uInt_meta[i] * sizeof(H),  // len in H-uint
            cudaMemcpyDeviceToHost);
    }
    // dump bit_meta and uInt_meta
    io::WriteBinaryFile(h_meta + n_chunk, (2 * n_chunk), new string(f_in + ".hmeta"));
    // write densely Huffman code and its metadata
    io::WriteBinaryFile(h, total_uInts, new string(f_in + ".dh"));
    // to save first, entry and keys
    io::WriteBinaryFile(                                    //
        reinterpret_cast<uint8_t*>(decode_meta),            //
        sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size,  // first, entry, reversed dict (keys)
        new string(f_in + ".cHcb"));

    size_t metadata_size = (2 * n_chunk) * sizeof(decltype(h_meta))              //
                           + sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size;  // uint8_t

    //////// clean up
    cudaFree(d_in);
    cudaFree(d_freq);
    cudaFree(d_canonical_cb);
    cudaFree(d_decode_meta);
    cudaFree(d_h);
    cudaFree(d_h_bitwidths);
    delete[] h;
    delete[] h_meta;
    delete[] decode_meta;

    return std::make_tuple(total_bits, total_uInts, metadata_size);
}

template <typename Q, typename H, typename DATA>
Q* HuffmanDecode(
    std::string& f_bcode_base,  //
    size_t       len,
    int          chunk_size,
    int          total_uInts,
    int          dict_size)
{
    auto type_bw             = sizeof(H) * 8;
    auto canonical_meta      = sizeof(H) * (2 * type_bw) + sizeof(Q) * dict_size;
    auto canonical_singleton = io::ReadBinaryFile<uint8_t>(f_bcode_base + ".cHcb", canonical_meta);
    cudaDeviceSynchronize();

    auto n_chunk  = (len - 1) / chunk_size + 1;
    auto hcode    = io::ReadBinaryFile<H>(f_bcode_base + ".dh", total_uInts);
    auto dH_meta  = io::ReadBinaryFile<size_t>(f_bcode_base + ".hmeta", 2 * n_chunk);
    auto blockDim = tBLK_DEFLATE;  // the same as deflating
    auto gridDim  = (n_chunk - 1) / blockDim + 1;

    auto d_xbcode              = mem::CreateCUDASpace<Q>(len);
    auto d_dHcode              = mem::CreateDeviceSpaceAndMemcpyFromHost(hcode, total_uInts);
    auto d_hcode_meta          = mem::CreateDeviceSpaceAndMemcpyFromHost(dH_meta, 2 * n_chunk);
    auto d_canonical_singleton = mem::CreateDeviceSpaceAndMemcpyFromHost(canonical_singleton, canonical_meta);
    cudaDeviceSynchronize();

    Decode<<<gridDim, blockDim, canonical_meta>>>(  //
        d_dHcode, d_hcode_meta, d_xbcode, len, chunk_size, n_chunk, d_canonical_singleton, (size_t)canonical_meta);
    cudaDeviceSynchronize();

    auto xbcode = mem::CreateHostSpaceAndMemcpyFromDevice(d_xbcode, len);
    cudaFree(d_xbcode);
    cudaFree(d_dHcode);
    cudaFree(d_hcode_meta);
    cudaFree(d_canonical_singleton);
    delete[] hcode;
    delete[] dH_meta;
    delete[] canonical_singleton;

    return xbcode;
}

template void wrapper::GetFrequency<uint8__t>(uint8__t*, size_t, unsigned int*, int);
template void wrapper::GetFrequency<uint16_t>(uint16_t*, size_t, unsigned int*, int);
template void wrapper::GetFrequency<uint32_t>(uint32_t*, size_t, unsigned int*, int);

template void PrintChunkHuffmanCoding<uint32_t>(size_t*, size_t*, size_t, int, size_t, size_t);
template void PrintChunkHuffmanCoding<uint64_t>(size_t*, size_t*, size_t, int, size_t, size_t);

template tuple3ul HuffmanEncode<uint8__t, uint32_t, float>(string&, uint8__t*, size_t, int, int);
template tuple3ul HuffmanEncode<uint16_t, uint32_t, float>(string&, uint16_t*, size_t, int, int);
template tuple3ul HuffmanEncode<uint32_t, uint32_t, float>(string&, uint32_t*, size_t, int, int);
template tuple3ul HuffmanEncode<uint8__t, uint64_t, float>(string&, uint8__t*, size_t, int, int);
template tuple3ul HuffmanEncode<uint16_t, uint64_t, float>(string&, uint16_t*, size_t, int, int);
template tuple3ul HuffmanEncode<uint32_t, uint64_t, float>(string&, uint32_t*, size_t, int, int);

template uint8__t* HuffmanDecode<uint8__t, uint32_t, float>(std::string&, size_t, int, int, int);
template uint16_t* HuffmanDecode<uint16_t, uint32_t, float>(std::string&, size_t, int, int, int);
template uint32_t* HuffmanDecode<uint32_t, uint32_t, float>(std::string&, size_t, int, int, int);
template uint8__t* HuffmanDecode<uint8__t, uint64_t, float>(std::string&, size_t, int, int, int);
template uint16_t* HuffmanDecode<uint16_t, uint64_t, float>(std::string&, size_t, int, int, int);
template uint32_t* HuffmanDecode<uint32_t, uint64_t, float>(std::string&, size_t, int, int, int);
// clang-format off
