#ifndef CUSZ_NVGPUSZ_CUH
#define CUSZ_NVGPUSZ_CUH

/**
 * @file nvgpusz.cuh
 * @author Jiannan Tian
 * @brief Workflow of cuSZ (header).
 * @version 0.3
 * @date 2021-07-12
 * (created) 2020-02-12 (rev.1) 2020-09-20 (rev.2) 2021-07-12; (rev.3) 2021-09-06
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cxxabi.h>
#include <clocale>
#include <iostream>
#include <unordered_map>

#include "capsule.hh"
#include "context.hh"
#include "header.hh"
#include "type_trait.hh"
#include "utils.hh"
#include "wrapper/extrap_lorenzo.cuh"
#include "wrapper/handle_sparsity.cuh"
// #include "wrapper/interp_spline_.h"

using namespace std;

template <typename T, typename E, typename H, typename FP>
class Compressor {
    using BYTE = uint8_t;

   private:
    struct {
        std::unordered_map<int, std::string> order2name = {
            {0, "header"},  {1, "book"},      {2, "quant"},         {3, "revbook"},
            {4, "outlier"}, {5, "huff-meta"}, {6, "huff-bitstream"}  //
        };

        std::unordered_map<std::string, uint32_t> nbyte = {
            {"header", sizeof(cusz_header)},
            {"book", 0U},            //
            {"quant", 0U},           //
            {"revbook", 0U},         //
            {"outlier", 0U},         //
            {"huff-meta", 0U},       //
            {"huff-bitstream", 0U},  //
        };
    } data_seg;

    // clang-format off
    struct { double eb; FP ebx2, ebx2_r, eb_r; } config;
    struct { float lossy{0.0}, outlier{0.0}, hist{0.0}, book{0.0}, lossless{0.0}; } time;
    // clang-format on

    struct {
        BYTE*   h_revbook;
        size_t *h_counts, *d_counts;
        H *     h_bitstream, *d_bitstream, *d_encspace;
    } huffman;

    struct {
        unsigned int workspace_nbyte;
        uint32_t     dump_nbyte;
        uint8_t*     workspace;
        uint8_t*     dump;
    } sp;

    cusz::PredictorLorenzo<T, E, FP>* predictor;
    cusz::OutlierHandler<T>*          csr;

    // context, configuration
    cuszCTX* ctx;
    //
    cusz_header* header;

    // OOD, indicated by v2
    dim3 xyz;

    unsigned int tune_deflate_chunksize(size_t len);
    void         report_compression_time();

    void consolidate(bool on_cpu = true, bool on_gpu = false);

    void lorenzo_dryrun(Capsule<T>* in_data);

    Compressor&
    get_freq_and_codebook(Capsule<E>* quant, Capsule<unsigned int>* freq, Capsule<H>* book, Capsule<uint8_t>* revbook);

    Compressor& analyze_compressibility(Capsule<unsigned int>* freq, Capsule<H>* book);

    Compressor& internal_eval_try_export_book(Capsule<H>* book);

    Compressor& internal_eval_try_export_quant(Capsule<E>* quant);

    // TODO make it return *this
    void try_skip_huffman(Capsule<E>* quant);

    Compressor& try_report_time();

    Compressor& huffman_encode(Capsule<E>* quant, Capsule<H>* book);

    Compressor& pack_metadata();

   public:
    Compressor(cuszCTX* _ctx);

    ~Compressor()
    {
        // release small-size arrays
        cudaFree(huffman.d_counts);
        cudaFree(huffman.d_bitstream);

        cudaFreeHost(huffman.h_bitstream);
        cudaFreeHost(huffman.h_counts);

        cudaFree(sp.workspace);
        cudaFreeHost(sp.dump);

        delete csr;
        delete predictor;
    };

    void compress(Capsule<T>* in_data);
};

template <typename T, typename E, typename H, typename FP>
class Decompressor {
   private:
    struct {
        std::unordered_map<std::string, int> name2order = {
            {"header", 0},  {"book", 1},      {"quant", 2},         {"revbook", 3},
            {"outlier", 4}, {"huff-meta", 5}, {"huff-bitstream", 6}  //
        };

        std::unordered_map<int, std::string> order2name = {
            {0, "header"},  {1, "book"},      {2, "quant"},         {3, "revbook"},
            {4, "outlier"}, {5, "huff-meta"}, {6, "huff-bitstream"}  //
        };

        std::unordered_map<std::string, uint32_t> nbyte = {
            {"header", sizeof(cusz_header)},
            {"book", 0U},            //
            {"quant", 0U},           //
            {"revbook", 0U},         //
            {"outlier", 0U},         //
            {"huff-meta", 0U},       //
            {"huff-bitstream", 0U},  //
        };
    } data_seg;

    std::vector<uint32_t> offsets;

    BYTE*  consolidated_dump;
    size_t cusza_nbyte;

    // clang-format off
    struct { float lossy{0.0}, outlier{0.0}, lossless{0.0}; } time;
    struct { double eb; FP ebx2, ebx2_r, eb_r; } config;

    struct { uint8_t *host, *dev; } csr_file;
    size_t m, mxm;
    // clang-format on

    cusz::OutlierHandler<T>*          csr;
    cusz::PredictorLorenzo<T, E, FP>* predictor;

    dim3         xyz;
    cuszCTX*     ctx;
    cusz_header* header;

    void try_report_decompression_time();

    void try_compare_with_origin(T* xdata);

    void try_write2disk(T* host_xdata);

    void huffman_decode(Capsule<E>* quant);

    void unpack_metadata();

   public:
    Decompressor(cuszCTX* ctx);

    ~Decompressor()
    {
        cudaFree(csr_file.dev);
        delete csr;
        delete predictor;
    }

    void decompress();
};

#endif
