//
// Created by jtian on 4/24/20.
//

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
#include "types.hh"

int ht_state_num;
int ht_all_nodes;
using uint8__t = uint8_t;

template <typename Q>
void wrapper::GetFrequency(Q* d_bcode, size_t len, unsigned int* d_freq, int dict_size)
{
    // Parameters for thread and block count optimization
    int maxbytes       = 98304;
    int numBuckets     = dict_size;
    int numValues      = len;
    int numSMs         = 84;  // set up parameters for V100
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
std::tuple<size_t, size_t, size_t> HuffmanEncode(string& f_bcode, Q* d_bcode, size_t len, int chunk_size, int dict_size)
{
    // prepare bcode
    //    auto bcode   = io::ReadBinaryFile<Q>(fname_bcode, len);
    //    auto d_bcode = mem::CreateDeviceSpaceAndMemcpyFromHost(bcode, len);

    // histogrammig
    ht_state_num = 2 * dict_size;
    ht_all_nodes = 2 * ht_state_num;
    auto d_freq  = mem::CreateCUDASpace<unsigned int>(ht_all_nodes);
    wrapper::GetFrequency(d_bcode, len, d_freq, dict_size);

    // get plain cb
    auto d_plain_cb = mem::CreateCUDASpace<H>(dict_size, 0xff);

    // wrapper::SetUpHuffmanTree<Q, H>(d_freq, d_plain_cb, dict_size);
    {
        InitHuffTreeAndGetCodebook<<<1, 32>>>(2 * dict_size, d_freq, d_plain_cb);
        cudaDeviceSynchronize();
    }

    // canonical Huffman; TODO should follow H to decide first and entry type
    auto type_bw = sizeof(H) * 8;
    // input, output, canonical codebook; numl, iterators, first, entry, reversed codebook
    auto total_bytes = sizeof(H) * (3 * dict_size) + sizeof(int) * (4 * type_bw) + sizeof(Q) * dict_size;

    auto d_singleton = mem::CreateCUDASpace<uint8__t>(total_bytes);

    // wrapper::MakeCanonical<Q, H>(d_plain_cb, d_singleton, total_bytes, dict_size);
    {
        auto d_input_cb = reinterpret_cast<H*>(d_singleton);
        cudaMemcpy(d_input_cb, d_plain_cb, sizeof(H) * dict_size, cudaMemcpyDeviceToDevice);
        void* args[] = {(void*)&d_singleton, (void*)&dict_size};
        cudaLaunchCooperativeKernel(                     // CUDA9 API
            (void*)GPU::GetCanonicalCode<H, Q>,          // kernel
            dim3((dict_size - 1) / tBLK_CANONICAL + 1),  // gridDim
            dim3(tBLK_CANONICAL),                        // blockDim
            args);
        cudaDeviceSynchronize();
    }
    auto singleton = mem::CreateHostSpaceAndMemcpyFromDevice(d_singleton, total_bytes);

    auto first = reinterpret_cast<int*>(singleton + sizeof(H) * (3 * dict_size)) + type_bw * 2;

    // coding by memcpy
    auto d_hcode        = mem::CreateCUDASpace<H>(len);
    auto d_canonical_cb = reinterpret_cast<H*>(d_singleton) + dict_size;

    {  // encode by memory copy of same-length code
        auto blockDim = tBLK_ENCODE;
        auto gridDim  = (len - 1) / blockDim + 1;
        EncodeFixedLen<Q, H><<<gridDim, blockDim>>>(d_bcode, d_hcode, len, d_canonical_cb);
        cudaDeviceSynchronize();
    }

    // deflating
    auto n_chunk           = (len - 1) / chunk_size + 1;
    auto d_hcode_bitwidths = mem::CreateCUDASpace<size_t>(n_chunk);

    {  // deflate
        auto blockDim = tBLK_DEFLATE;
        auto gridDim  = (n_chunk - 1) / blockDim + 1;
        Deflate<H><<<gridDim, blockDim>>>(d_hcode, len, d_hcode_bitwidths, chunk_size);
        cudaDeviceSynchronize();
    }

    // dump TODO change to int
    auto hcode_meta    = new size_t[n_chunk * 3]();
    auto dH_uInt_meta  = hcode_meta;
    auto dH_bit_meta   = hcode_meta + n_chunk;
    auto dH_uInt_entry = hcode_meta + n_chunk * 2;
    // copy back densely Huffman code (dHcode)
    cudaMemcpy(dH_bit_meta, d_hcode_bitwidths, n_chunk * sizeof(size_t), cudaMemcpyDeviceToHost);
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
    printf("Huffman bitstream: %lu chunks of size = %d, in %lu uint%lus or %lu bits\n", n_chunk, chunk_size, total_uInts, sizeof(H) * 8, total_bits);

    // print densely metadata
    PrintChunkHuffmanCoding<H>(dH_bit_meta, dH_uInt_meta, len, chunk_size, total_bits, total_uInts);

    // copy back densely Huffman code in units of uInt (regarding endianness)
    auto hcode = new H[total_uInts]();
    for (auto i = 0; i < n_chunk; i++) {
        cudaMemcpy(
            hcode + dH_uInt_entry[i],     // dst
            d_hcode + i * chunk_size,     // src
            dH_uInt_meta[i] * sizeof(H),  // len in H-uint
            cudaMemcpyDeviceToHost);
    }
    // dump bit_meta and uInt_meta
    io::WriteBinaryFile(hcode_meta + n_chunk, (2 * n_chunk), new string(f_bcode + ".hmeta"));
    // write densely Huffman code and its metadata
    io::WriteBinaryFile(hcode, total_uInts, new string(f_bcode + ".dh"));
    // to save first, entry and keys
    io::WriteBinaryFile(                                      //
        reinterpret_cast<uint8__t*>(first),                   //
        sizeof(int) * (2 * type_bw) + sizeof(Q) * dict_size,  // first, entry, reversed dict (keys)
        new string(f_bcode + ".cHcb"));

    size_t metadata_size = (2 * n_chunk) * sizeof(decltype(hcode_meta))            //
                           + sizeof(int) * (2 * type_bw) + sizeof(Q) * dict_size;  // uint8__t

    cudaFree(d_bcode);
    cudaFree(d_freq);
    cudaFree(d_plain_cb);
    cudaFree(d_hcode);
    cudaFree(d_hcode_bitwidths);
    delete[] hcode;
    delete[] hcode_meta;

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
    auto canonical_meta      = sizeof(int) * (2 * type_bw) + sizeof(Q) * dict_size;
    auto canonical_singleton = io::ReadBinaryFile<uint8__t>(f_bcode_base + ".cHcb", canonical_meta);

    auto n_chunk  = (len - 1) / chunk_size + 1;
    auto hcode    = io::ReadBinaryFile<H>(f_bcode_base + ".dh", total_uInts);
    auto dH_meta  = io::ReadBinaryFile<size_t>(f_bcode_base + ".hmeta", 2 * n_chunk);
    auto blockDim = tBLK_DEFLATE;  // the same as deflating
    auto gridDim  = (n_chunk - 1) / blockDim + 1;

    auto d_xbcode              = mem::CreateCUDASpace<Q>(len);
    auto d_dHcode              = mem::CreateDeviceSpaceAndMemcpyFromHost(hcode, total_uInts);
    auto d_hcode_meta          = mem::CreateDeviceSpaceAndMemcpyFromHost(dH_meta, 2 * n_chunk);
    auto d_canonical_singleton = mem::CreateDeviceSpaceAndMemcpyFromHost(canonical_singleton, canonical_meta);

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

    return xbcode;
}

template void wrapper::GetFrequency<uint8__t>(uint8__t* d_bcode, size_t len, unsigned int* d_freq, int dict_size);
template void wrapper::GetFrequency<uint16_t>(uint16_t* d_bcode, size_t len, unsigned int* d_freq, int dict_size);
template void wrapper::GetFrequency<uint32_t>(uint32_t* d_bcode, size_t len, unsigned int* d_freq, int dict_size);

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
