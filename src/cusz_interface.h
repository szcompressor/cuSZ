#ifndef CUSZ_INTERFACE_H
#define CUSZ_INTERFACE_H

/**
 * @file cusz_interface.h
 * @author Jiannan Tian
 * @brief Workflow of cuSZ (header).
 * @version 0.3
 * @date 2021-07-12
 * (created) 2020-02-12 (rev.1) 2020-09-20 (rev.2) 2021-07-12
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <iostream>
#include "argparse.hh"
#include "datapack.hh"
#include "kernel/dryrun.h"
#include "pack.hh"
#include "type_trait.hh"
#include "utils.hh"
#include "wrapper/extrap_lorenzo.h"
#include "wrapper/handle_sparsity.h"
#include "wrapper/huffman_enc_dec.cuh"
#include "wrapper/huffman_parbook.cuh"

using namespace std;

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz_compress(
    argpack*,
    struct PartialData<typename DataTrait<If_FP, DataByte>::Data>*,
    dim3,
    metadata_pack* mp,
    unsigned int = 1);

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz_decompress(argpack*, metadata_pack* mp);

template <typename Data, typename Quant, typename Huff, typename FP>
class Compressor {
   private:
    static const auto TYPE_BITCOUNT = sizeof(Huff) * 8;

    unsigned int tune_deflate_chunksize(size_t len)
    {
        cout << "autotuning deflate chunksize\n";

        int current_dev = 0;
        cudaSetDevice(current_dev);
        cudaDeviceProp dev_prop{};
        cudaGetDeviceProperties(&dev_prop, current_dev);

        auto nSM                = dev_prop.multiProcessorCount;
        auto allowed_block_dim  = dev_prop.maxThreadsPerBlock;
        auto deflate_nthread    = allowed_block_dim * nSM / HuffConfig::deflate_constant;
        auto optimal_chunk_size = (len + deflate_nthread - 1) / deflate_nthread;
        optimal_chunk_size      = ((optimal_chunk_size - 1) / HuffConfig::Db_deflate + 1) * HuffConfig::Db_deflate;

        return optimal_chunk_size;
    }

    void report_compression_time(size_t len, float lossy, float outlier, float hist, float book, float lossless)
    {
        auto display_throughput = [](float time, size_t nbyte) {
            auto throughput = nbyte * 1.0 / (1024 * 1024 * 1024) / (time * 1e-3);
            cout << throughput << "GiB/s\n";
        };
        //
        cout << "\nTIME in milliseconds\t================================================================\n";
        float nonbook = lossy + outlier + hist + lossless;

        printf("TIME\tconstruct:\t%f\t", lossy), display_throughput(lossy, len * sizeof(Data));
        printf("TIME\toutlier:\t%f\t", outlier), display_throughput(outlier, len * sizeof(Data));
        printf("TIME\thistogram:\t%f\t", hist), display_throughput(hist, len * sizeof(Data));
        printf("TIME\tencode:\t%f\t", lossless), display_throughput(lossless, len * sizeof(Data));

        cout << "TIME\t--------------------------------------------------------------------------------\n";
        printf("TIME\tnon-book kernels (sum):\t%f\t", nonbook), display_throughput(nonbook, len * sizeof(Data));
        cout << "TIME\t================================================================================\n";
        printf("TIME\tbuild book (not counted in prev section):\t%f\t", book),
            display_throughput(book, len * sizeof(Data));
        printf("TIME\t*all* kernels (sum, count book time):\t%f\t", nonbook + book),
            display_throughput(nonbook + book, len * sizeof(Data));
        cout << "TIME\t================================================================================\n\n";
    }

    void export_codebook(Huff* d_book, const string& basename, size_t dict_size)
    {
        auto              h_book = mem::create_devspace_memcpy_d2h(d_book, dict_size);
        std::stringstream s;
        s << basename + "-" << dict_size << "-ui" << sizeof(Huff) << ".lean-book";
        logging(log_dbg, "export \"lean\" codebook (of dict_size) as", s.str());
        io::write_array_to_binary(s.str(), h_book, dict_size);
        cudaFreeHost(h_book);
        h_book = nullptr;
    }

   public:
    struct {
        unsigned int data;
        unsigned int quant;
        unsigned int anchor;
        int          nnz_outlier;  // TODO modify the type correspondingly
        unsigned int dict_size;
    } length;

    int ndim;

    struct {
        int    radius;
        double eb;
        FP     ebx2;
        FP     ebx2_r;
        FP     eb_r;
    } config;

    struct {
        float lossy, outlier, hist, book, lossless;
    } time;

    struct {
        size_t num_bits, num_uints, revbook_nbyte;
    } huffman_meta;

    argpack* ap;

    unsigned int get_revbook_nbyte() { return sizeof(Huff) * (2 * TYPE_BITCOUNT) + sizeof(Quant) * length.dict_size; }

    Compressor(argpack* _ap, unsigned int _data_len, double _eb)
    {
        ap = _ap;

        ndim = ap->ndim;

        config.radius = ap->radius;

        length.data      = _data_len;
        length.quant     = length.data;  // TODO if lorenzo
        length.dict_size = ap->dict_size;

        config.eb     = _eb;
        config.ebx2   = _eb * 2;
        config.ebx2_r = 1 / (_eb * 2);
        config.eb_r   = 1 / _eb;

        if (ap->sz_workflow.autotune_huffchunk) ap->huffman_chunk = tune_deflate_chunksize(length.data);
    }

    void lorenzo_dryrun(struct PartialData<Data>* in_data)
    {
        auto get_npart = [](auto size, auto subsize) { return (size + subsize - 1) / subsize; };

        if (ap->sz_workflow.dryrun) {
            auto len    = length.data;
            auto eb     = config.eb;
            auto ebx2_r = config.ebx2_r;
            auto ebx2   = config.ebx2;

            logging(log_info, "invoke dry-run");
            constexpr auto SEQ       = 4;
            constexpr auto SUBSIZE   = 256;
            auto           dim_block = SUBSIZE / SEQ;
            auto           dim_grid  = get_npart(len, SUBSIZE);

            cusz::dual_quant_dryrun<Data, float, SUBSIZE, SEQ>
                <<<dim_grid, dim_block>>>(in_data->dptr, len, ebx2_r, ebx2);
            HANDLE_ERROR(cudaDeviceSynchronize());

            Data* dryrun_result;
            cudaMallocHost(&dryrun_result, len * sizeof(Data));
            cudaMemcpy(dryrun_result, in_data->dptr, len * sizeof(Data), cudaMemcpyDeviceToHost);

            analysis::verify_data<Data>(&ap->stat, dryrun_result, in_data->hptr, len);
            analysis::print_data_quality_metrics<Data>(&ap->stat, false, eb, 0);

            cudaFreeHost(dryrun_result);

            exit(0);
        }

        // return *this;
    }

    Compressor& predict_quantize(struct PartialData<Data>* data, dim3 xyz, struct PartialData<Quant>* quant)
    {
        logging(log_info, "invoke lossy-construction");
        if (ap->sz_workflow.predictor == "lorenzo") {
            compress_lorenzo_construct<Data, Quant, float>(
                data->dptr, quant->dptr, xyz, ap->ndim, config.eb, config.radius, time.lossy);
        }
        else if (ap->sz_workflow.predictor == "spline3d") {
            throw std::runtime_error("spline not impl'ed");
            if (ap->ndim != 3) throw std::runtime_error("Spline3D must be for 3D data.");
            // compress_spline3d_construct<Data, Quant, float>(
            //     in_data->dptr, quant.dptr, xyz, ap->ndim, eb, radius, time_lossy);
        }
        else {
            throw std::runtime_error("need to specify predcitor");
        }

        return *this;
    }

    Compressor& gather_outlier(struct PartialData<Data>* in_data)
    {
        unsigned int workspace_nbyte, dump_nbyte;
        uint8_t *    workspace, *dump;
        workspace_nbyte = get_compress_sparse_workspace<Data>(length.data);
        cudaMalloc((void**)&workspace, workspace_nbyte);
        cudaMallocHost((void**)&dump, workspace_nbyte);

        OutlierHandler<Data> csr(length.data);
        csr.configure(workspace)  //
            .gather_CUDA10(in_data->dptr, dump_nbyte, time.outlier)
            .archive(dump, length.nnz_outlier);
        io::write_array_to_binary(ap->subfiles.compress.out_outlier, dump, dump_nbyte);

        cudaFree(workspace), cudaFreeHost(dump);

        auto fmt_nnz = "(" + std::to_string(length.nnz_outlier / 1.0 / length.data * 100) + "%)";
        logging(log_info, "nnz/#outlier:", length.nnz_outlier, fmt_nnz, "saved");

        return *this;
    }

    Compressor& get_freq_and_codebook(
        struct PartialData<Quant>*        quant,
        struct PartialData<unsigned int>* freq,
        struct PartialData<Huff>*         book,
        struct PartialData<uint8_t>*      revbook)
    {
        wrapper::get_frequency<Quant>(quant->dptr, length.quant, freq->dptr, length.dict_size, time.hist);

        {  // This is end-to-end time for parbook.
            auto t = new cuda_timer_t;
            t->timer_start();
            lossless::par_get_codebook<Quant, Huff>(length.dict_size, freq->dptr, book->dptr, revbook->dptr);
            time.book = t->timer_end_get_elapsed_time();
            cudaDeviceSynchronize();
            delete t;
        }

        return *this;
    }

    Compressor& analyze_compressibility(
        struct PartialData<unsigned int>* freq,  //
        struct PartialData<Huff>*         book)
    {
        if (ap->report.compressibility) {
            cudaMallocHost(&freq->hptr, freq->nbyte()), freq->d2h();
            cudaMallocHost(&book->hptr, book->nbyte()), book->d2h();

            Analyzer analyzer{};
            analyzer  //
                .EstimateFromHistogram(freq->hptr, length.dict_size)
                .template GetHuffmanCodebookStat<Huff>(freq->hptr, book->hptr, length.data, length.dict_size)
                .PrintCompressibilityInfo(true);

            cudaFreeHost(freq->hptr);
            cudaFreeHost(book->hptr);
        }

        return *this;
    }

    Compressor& internal_eval_try_export_book(struct PartialData<Huff>* book)
    {
        // internal evaluation, not stored in sz archive
        if (ap->sz_workflow.export_book) {  //
            export_codebook(book->dptr, ap->subfiles.compress.huff_base, length.dict_size);
            logging(log_info, "exporting codebook as binary; suffix: \".lean-book\"");
        }
        return *this;
    }

    Compressor& internal_eval_try_export_quant(struct PartialData<Quant>* quant)
    {
        // internal_eval
        if (ap->sz_workflow.export_quant) {  //
            cudaMallocHost(&quant->hptr, quant->nbyte());
            quant->d2h();

            io::write_array_to_binary(ap->subfiles.compress.raw_quant, quant->hptr, length.quant);
            logging(log_info, "exporting quant as binary; suffix: \".lean-quant\"");
            logging(log_info, "exiting");
            exit(0);
        }
        return *this;
    }

    // TODO make it return *this
    void try_skip_huffman(struct PartialData<Quant>* quant)
    {
        // decide if skipping Huffman coding
        if (ap->sz_workflow.skip_huffman) {
            cudaMallocHost(&quant->hptr, quant->nbyte());
            quant->d2h();

            io::write_array_to_binary(ap->subfiles.compress.out_quant, quant->hptr, length.quant);
            logging(log_info, "to store quant.code directly (Huffman enc skipped)");
            exit(0);
        }
    }

    Compressor& try_report_time()
    {
        if (ap->report.time)
            report_compression_time(length.data, time.lossy, time.outlier, time.hist, time.book, time.lossless);
        return *this;
    }

    Compressor& export_revbook(struct PartialData<uint8_t>* revbook)
    {
        revbook->d2h();
        io::write_array_to_binary(ap->subfiles.compress.huff_base + ".canon", revbook->hptr, get_revbook_nbyte());
        cudaFreeHost(revbook->hptr);

        return *this;
    }

    Compressor& huffman_encode(
        struct PartialData<Quant>* quant,  //
        struct PartialData<Huff>*  book)
    {
        lossless::HuffmanEncode<Quant, Huff>(
            ap->subfiles.compress.huff_base, quant->dptr, book->dptr, length.quant, ap->huffman_chunk, ap->dict_size,
            huffman_meta.num_bits, huffman_meta.num_uints, time.lossless);

        huffman_meta.revbook_nbyte = get_revbook_nbyte();

        return *this;
    }

    //
};

template class Compressor<float, uint8_t, uint32_t, float>;
template class Compressor<float, uint16_t, uint32_t, float>;
template class Compressor<float, uint32_t, uint32_t, float>;
template class Compressor<float, uint8_t, unsigned long long, float>;
template class Compressor<float, uint16_t, unsigned long long, float>;
template class Compressor<float, uint32_t, unsigned long long, float>;

class Decompressor {
   public:
    argpack* ap;
};

#endif
