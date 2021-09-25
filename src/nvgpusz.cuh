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
#include "wrapper/huffman_coarse.cuh"

using namespace std;

typedef struct DataSegmentDescription {
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

    std::vector<uint32_t> offset;

    uint32_t get_offset(std::string name) { return offset.at(name2order.at(name)); }

} DataSeg;

template <typename T, typename E, typename H, typename FP>
class Compressor {
    using BYTE = uint8_t;

   private:
    DataSeg dataseg;

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
    cusz::HuffmanWork<E, H>*          reducer;

    cuszCTX*     ctx;
    cusz_header* header;

    cusz::WHEN timing;

    size_t cusza_nbyte;
    size_t m, mxm;

    struct {
        uint8_t *host, *dev;
    } csr_file;

    BYTE* dump;

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

    void try_report_decompression_time();

    void try_compare_with_origin(T* xdata);

    void try_write2disk(T* host_xdata);

    void huffman_decode(Capsule<E>* quant);

    void unpack_metadata();

   public:
    Compressor(cuszCTX* _ctx, cusz::WHEN _timing) : ctx(_ctx)
    {
        timing = _timing;

        if (timing == cusz::WHEN::COMPRESS or    //
            timing == cusz::WHEN::EXPERIMENT or  //
            timing == cusz::WHEN::COMPRESS_DRYRUN) {
            header = new cusz_header();

            ctx->quant_len = ctx->data_len;  // TODO if lorenzo

            ConfigHelper::set_eb_series(ctx->eb, config);

            if (ctx->on_off.autotune_huffchunk) ctx->huffman_chunk = tune_deflate_chunksize(ctx->data_len);

            csr = new cusz::OutlierHandler<T>(ctx->data_len, &sp.workspace_nbyte);
            // can be known on Compressor init
            cudaMalloc((void**)&sp.workspace, sp.workspace_nbyte);
            cudaMallocHost((void**)&sp.dump, sp.workspace_nbyte);

            xyz = dim3(ctx->x, ctx->y, ctx->z);

            predictor = new cusz::PredictorLorenzo<T, E, FP>(xyz, ctx->eb, ctx->radius, false);

            ctx->quant_len = predictor->get_quant_len();
        }
        else if (timing == cusz::WHEN::DECOMPRESS) {
            auto fname_dump = ctx->fnames.path2file + ".cusza";
            cusza_nbyte     = ConfigHelper::get_filesize(fname_dump);
            dump            = io::read_binary_to_new_array<BYTE>(fname_dump, cusza_nbyte);
            header          = reinterpret_cast<cusz_header*>(dump);

            unpack_metadata();

            m   = Reinterpret1DTo2D::get_square_size(ctx->data_len);
            mxm = m * m;

            xyz = dim3(header->x, header->y, header->z);

            csr           = new cusz::OutlierHandler<T>(ctx->data_len, ctx->nnz_outlier);
            csr_file.host = reinterpret_cast<BYTE*>(dump + dataseg.get_offset("outlier"));
            cudaMalloc((void**)&csr_file.dev, csr->get_total_nbyte());
            cudaMemcpy(csr_file.dev, csr_file.host, csr->get_total_nbyte(), cudaMemcpyHostToDevice);

            predictor = new cusz::PredictorLorenzo<T, E, FP>(xyz, ctx->eb, ctx->radius, false);

            reducer = new cusz::HuffmanWork<E, H>(
                header->quant_len, dump,  //
                header->huffman_chunk, header->huffman_num_uints, header->dict_size);

            LOGGING(LOG_INFO, "decompressing...");
        }
    }

    ~Compressor()
    {
        if (timing == cusz::WHEN::COMPRESS) {  // release small-size arrays
            cudaFree(huffman.d_counts);
            cudaFree(huffman.d_bitstream);

            cudaFreeHost(huffman.h_bitstream);
            cudaFreeHost(huffman.h_counts);

            cudaFree(sp.workspace);
            cudaFreeHost(sp.dump);
        }
        else {
            cudaFree(csr_file.dev);
        }

        delete csr;
        delete predictor;
    };

    void compress(Capsule<T>* in_data);

    void decompress();
};

#endif
