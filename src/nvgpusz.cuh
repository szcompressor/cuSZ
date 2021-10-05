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

#include "common/capsule.hh"
#include "common/type_traits.hh"
#include "context.hh"
#include "header.hh"
#include "utils.hh"
#include "wrapper/extrap_lorenzo.cuh"
#include "wrapper/handle_sparsity10.cuh"
#include "wrapper/handle_sparsity11.cuh"
#include "wrapper/huffman_coarse.cuh"

using namespace std;

template <typename T, typename E, typename H, typename FP>
class Compressor {
    using BYTE = uint8_t;

   private:
    DataSeg dataseg;

    // clang-format off
    struct { double eb; FP ebx2, ebx2_r, eb_r; } config;
    struct { float lossy{0.0}, outlier{0.0}, hist{0.0}, book{0.0}, lossless{0.0}; } time;
    // clang-format on

    uint32_t sp_dump_nbyte;

    // worker
    cusz::PredictorLorenzo<T, E, FP>* predictor;
    cusz::OutlierHandler10<T>*        csr;
    cusz::HuffmanWork<E, H>*          reducer;

    // data fields
    Capsule<T>          anchor;
    Capsule<E>          quant;
    Capsule<cusz::FREQ> freq;
    Capsule<H>          book;
    Capsule<H>          huff_data;
    Capsule<size_t>     huff_counts;
    Capsule<BYTE>       revbook;
    Capsule<BYTE>       sp_use;

    cuszCTX*     ctx;
    cusz_header* header;

    Capsule<T>*    in_data;  // compress-time, TODO rename
    Capsule<BYTE>* in_dump;  // decompress-time, TODO rename

    cuszWHEN timing;

    // TODO MetadataT
    // size_t cusza_nbyte;
    size_t m, mxm;

    dim3 xyz;

    uint32_t tune_deflate_chunksize(size_t len);

    Compressor& lorenzo_dryrun(Capsule<T>* in_data);

    Compressor& prescan();

    Compressor& analyze_compressibility(Capsule<cusz::FREQ>* freq, Capsule<H>* book);
    Compressor& internal_eval_try_export_book(Capsule<H>* book);
    Compressor& internal_eval_try_export_quant(Capsule<E>* quant);

    Compressor& try_report_compress_time();
    Compressor& try_report_decompress_time();

    Compressor& try_skip_huffman(Capsule<E>* quant);
    Compressor& try_compare_with_origin(T* xdata);
    Compressor& try_write2disk(T* host_xdata);

    Compressor&
    get_freq_codebook(Capsule<E>* quant, Capsule<cusz::FREQ>* freq, Capsule<H>* book, Capsule<uint8_t>* revbook);
    Compressor& huffman_encode(Capsule<E>* quant, Capsule<H>* book);
    // Compressor& huffman_decode(Capsule<E>* quant); // done encapsulation

    Compressor& pack_metadata();
    Compressor& unpack_metadata();

   public:
    uint32_t get_decompress_space_len() { return mxm + ChunkingTrait<1>::BLOCK; }

   public:
    Compressor(cuszCTX* _ctx, Capsule<T>* _in_data);

    Compressor(cuszCTX* _ctx, Capsule<BYTE>* _in_dump);  // TODO excl. T == BYTE

    ~Compressor();

    Compressor& compress();

    template <cuszLOC SRC, cuszLOC DST>
    Compressor& consolidate(BYTE** dump);

    Compressor& decompress(Capsule<T>* out_xdata);
    Compressor& backmatter(Capsule<T>* out_xdata);
};

#endif
