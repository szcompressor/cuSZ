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

#include "argparse.hh"
#include "capsule.hh"
#include "header.hh"
#include "type_trait.hh"
#include "utils.hh"
#include "wrapper/handle_sparsity.cuh"
#include "wrapper/interp_spline_.h"

using namespace std;

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz_compress(
    argpack*,
    Capsule<typename DataTrait<If_FP, DataByte>::Data>*,
    dim3,
    cusz_header* mp,
    unsigned int = 1);

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz_decompress(argpack*, cusz_header* mp);

template <typename Data, typename Quant, typename Huff, typename FP>
class Compressor {
    using BYTE = uint8_t;

   private:
    struct {
        std::unordered_map<int, std::string> order2name = {
            {0, "header"},  {1, "book"},      {2, "quant"},         {3, "revbook"},
            {4, "outlier"}, {5, "huff-meta"}, {6, "huff-bitstream"}  //
        };

        std::unordered_map<std::string, uint32_t> nbyte_raw = {
            {"header", sizeof(cusz_header)},
            {"book", 0U},            //
            {"quant", 0U},           //
            {"revbook", 0U},         //
            {"outlier", 0U},         //
            {"huff-meta", 0U},       //
            {"huff-bitstream", 0U},  //
        };
    } data_seg;

    static const auto TYPE_BITCOUNT = sizeof(Huff) * 8;

    unsigned int tune_deflate_chunksize(size_t len);

    void report_compression_time(size_t len, float lossy, float outlier, float hist, float book, float lossless);

   public:
    Spline3<Data*, Quant*, float>* spline3;

    void register_spline3(Spline3<Data*, Quant*, float>* _spline3) { spline3 = _spline3; }
    struct {
        unsigned int data, quant, anchor;
        int          nnz_outlier;  // TODO modify the type correspondingly
        unsigned int dict_size;
    } length;

    int ndim;

    struct {
        int    radius;
        double eb;
        FP     ebx2, ebx2_r, eb_r;
    } config;

    struct {
        float lossy{0.0}, outlier{0.0}, hist{0.0}, book{0.0}, lossless{0.0};
    } time;

    struct {
        struct {
            size_t num_bits, num_uints, revbook_nbyte;
        } meta;

        struct {
            BYTE*   h_revbook;
            size_t *h_counts, *d_counts;
            Huff *  h_bitstream, *d_bitstream, *d_encspace;
        } array;
    } huffman;

    struct {
        unsigned int workspace_nbyte;
        uint32_t     dump_nbyte;
        uint8_t*     workspace;
        uint8_t*     dump;
    } sp;

    OutlierHandler<Data>* csr;

    // context, configuration
    argpack* ctx;
    //
    cusz_header* header;

    unsigned int get_revbook_nbyte() { return sizeof(Huff) * (2 * TYPE_BITCOUNT) + sizeof(Quant) * length.dict_size; }

    Compressor(argpack* _ctx);

    ~Compressor()
    {
        // release small-size arrays
        cudaFree(huffman.array.d_counts);
        cudaFree(huffman.array.d_bitstream);

        cudaFreeHost(huffman.array.h_bitstream);
        cudaFreeHost(huffman.array.h_counts);

        cudaFreeHost(sp.dump);

        delete csr;
    };

    void consolidate(bool on_cpu = true, bool on_gpu = false)
    {
        // put in header
        header->nbyte.book           = data_seg.nbyte_raw.at("book");
        header->nbyte.revbook        = data_seg.nbyte_raw.at("revbook");
        header->nbyte.outlier        = data_seg.nbyte_raw.at("outlier");
        header->nbyte.huff_meta      = data_seg.nbyte_raw.at("huff-meta");
        header->nbyte.huff_bitstream = data_seg.nbyte_raw.at("huff-bitstream");

        // consolidate
        std::vector<uint32_t> offsets = {0};

        printf(
            "\ndata segments:\n  \e[1m\e[31m%-18s\t%12s\t%15s\t%15s\e[0m\n",  //
            const_cast<char*>("name"),                                        //
            const_cast<char*>("nbyte"),                                       //
            const_cast<char*>("start"),                                       //
            const_cast<char*>("end"));

        // print long numbers with thousand separator
        // https://stackoverflow.com/a/7455282
        // https://stackoverflow.com/a/11695246
        setlocale(LC_ALL, "");

        for (auto i = 0; i < 7; i++) {
            const auto& name = data_seg.order2name.at(i);

            auto o = offsets.back() + __cusz_get_alignable_len<BYTE, 128>(data_seg.nbyte_raw.at(name));
            offsets.push_back(o);

            printf(
                "  %-18s\t%'12u\t%'15u\t%'15u\n", name.c_str(), data_seg.nbyte_raw.at(name), offsets.at(i),
                offsets.back());
        }

        auto total_nbyte = offsets.back();

        printf("\ncompression ratio:\t%.4f\n", ctx->data_len * sizeof(Data) * 1.0 / total_nbyte);

        BYTE* h_dump = nullptr;
        // BYTE* d_dump = nullptr;

        cout << "dump on CPU\t" << on_cpu << '\n';
        cout << "dump on GPU\t" << on_gpu << '\n';

        auto both = on_cpu and on_gpu;
        if (both) {
            //
            throw runtime_error("[consolidate on both] not implemented");
        }
        else {
            if (on_cpu) {
                //
                cudaMallocHost(&h_dump, total_nbyte);

                /* 0 */  // header
                cudaMemcpy(
                    h_dump + offsets.at(0),           //
                    reinterpret_cast<BYTE*>(header),  //
                    data_seg.nbyte_raw.at("header"),  //
                    cudaMemcpyHostToHost);
                /* 1 */  // book
                /* 2 */  // quant
                /* 3 */  // revbook
                cudaMemcpy(
                    h_dump + offsets.at(3),                            //
                    reinterpret_cast<BYTE*>(huffman.array.h_revbook),  //
                    data_seg.nbyte_raw.at("revbook"),                  //
                    cudaMemcpyHostToHost);
                /* 4 */  // outlier
                cudaMemcpy(
                    h_dump + offsets.at(4),            //
                    reinterpret_cast<BYTE*>(sp.dump),  //
                    data_seg.nbyte_raw.at("outlier"),  //
                    cudaMemcpyHostToHost);
                /* 5 */  // huff_meta
                cudaMemcpy(
                    h_dump + offsets.at(5),                                         //
                    reinterpret_cast<BYTE*>(huffman.array.h_counts + ctx->nchunk),  //
                    data_seg.nbyte_raw.at("huff-meta"),                             //
                    cudaMemcpyHostToHost);
                /* 6 */  // huff_bitstream
                cudaMemcpy(
                    h_dump + offsets.at(6),                              //
                    reinterpret_cast<BYTE*>(huffman.array.h_bitstream),  //
                    data_seg.nbyte_raw.at("huff-bitstream"),             //
                    cudaMemcpyHostToHost);

                auto output_name = ctx->fnames.path_basename + ".cusza";
                cout << "output:\t" << output_name << '\n';

                io::write_array_to_binary(output_name, h_dump, total_nbyte);

                cudaFreeHost(h_dump);
            }
            else {
                throw runtime_error("[consolidate on both] not implemented");
            }
        }

        //
    };

    void lorenzo_dryrun(Capsule<Data>* in_data);

    Compressor& predict_quantize(Capsule<Data>* data, dim3 xyz, Capsule<Data>* anchor, Capsule<Quant>* quant);

    Compressor& gather_outlier(Capsule<Data>* in_data);

    Compressor& get_freq_and_codebook(
        Capsule<Quant>*        quant,
        Capsule<unsigned int>* freq,
        Capsule<Huff>*         book,
        Capsule<uint8_t>*      revbook);

    Compressor& analyze_compressibility(Capsule<unsigned int>* freq, Capsule<Huff>* book);

    Compressor& internal_eval_try_export_book(Capsule<Huff>* book);

    Compressor& internal_eval_try_export_quant(Capsule<Quant>* quant);

    // TODO make it return *this
    void try_skip_huffman(Capsule<Quant>* quant);

    Compressor& try_report_time();

    Compressor& export_revbook(Capsule<uint8_t>* revbook);

    Compressor& huffman_encode(Capsule<Quant>* quant, Capsule<Huff>* book);

    Compressor& pack_metadata();
};

template <typename Data, typename Quant, typename Huff, typename FP>
class Decompressor {
   private:
    void report_decompression_time(size_t len, float lossy, float outlier, float lossless);

    void unpack_metadata();

    struct {
        std::unordered_map<std::string, int> name2order = {
            {"header", 0},  {"book", 1},      {"quant", 2},         {"revbook", 3},
            {"outlier", 4}, {"huff-meta", 5}, {"huff-bitstream", 6}  //
        };

        std::unordered_map<int, std::string> order2name = {
            {0, "header"},  {1, "book"},      {2, "quant"},         {3, "revbook"},
            {4, "outlier"}, {5, "huff-meta"}, {6, "huff-bitstream"}  //
        };

        std::unordered_map<std::string, uint32_t> nbyte_raw = {
            {"header", sizeof(cusz_header)},
            {"book", 0U},            //
            {"quant", 0U},           //
            {"revbook", 0U},         //
            {"outlier", 0U},         //
            {"huff-meta", 0U},       //
            {"huff-bitstream", 0U},  //
        };
    } data_seg;

   public:
    void register_spline3(Spline3<Data*, Quant*, float>* _spline3) { spline3 = _spline3; }

    Spline3<Data*, Quant*, FP>* spline3;

    dim3 xyz;

    struct {
        BYTE* whole;
        // cusz_header* header;
    } consolidated_dump;

    void read_array_nbyte_from_header()
    {
        data_seg.nbyte_raw.at("book")           = header->nbyte.book;
        data_seg.nbyte_raw.at("revbook")        = header->nbyte.revbook;
        data_seg.nbyte_raw.at("outlier")        = header->nbyte.outlier;
        data_seg.nbyte_raw.at("huff-meta")      = header->nbyte.huff_meta;
        data_seg.nbyte_raw.at("huff-bitstream") = header->nbyte.huff_bitstream;

        // cout << "nbyte_raw.at(book)           = " << header->nbyte.book << '\n';
        // cout << "nbyte_raw.at(revbook)        = " << header->nbyte.revbook << '\n';
        // cout << "nbyte_raw.at(outlier)        = " << header->nbyte.outlier << '\n';
        // cout << "nbyte_raw.at(huff-meta)      = " << header->nbyte.huff_meta << '\n';
        // cout << "nbyte_raw.at(huff-bitstream) = " << header->nbyte.huff_bitstream << '\n';
    }

    std::vector<uint32_t> offsets;

    void get_data_seg_offsets()
    {
        /* 0 header */ offsets.push_back(0);

        if (ctx->verbose) {
            printf(
                "\ndata segments (verification):\n  \e[1m\e[31m%-18s\t%12s\t%15s\t%15s\e[0m\n",  //
                const_cast<char*>("name"),                                                       //
                const_cast<char*>("nbyte"),                                                      //
                const_cast<char*>("start"),                                                      //
                const_cast<char*>("end"));

            setlocale(LC_ALL, "");
        }

        for (auto i = 0; i < 7; i++) {
            const auto& name  = data_seg.order2name.at(i);
            auto        nbyte = data_seg.nbyte_raw.at(name);
            auto        o     = offsets.back() + __cusz_get_alignable_len<BYTE, 128>(nbyte);
            offsets.push_back(o);

            if (ctx->verbose) {
                printf(
                    "  %-18s\t%'12u\t%'15u\t%'15u\n", name.c_str(), data_seg.nbyte_raw.at(name), offsets.at(i),
                    offsets.back());
            }
        }
    }

    size_t cusza_nbyte;

    struct {
        float lossy{0.0}, outlier{0.0}, lossless{0.0};
    } time;
    struct {
        unsigned int data, quant, anchor;
        int          nnz_outlier;  // TODO modify the type correspondingly
        unsigned int dict_size;
    } length;

    struct {
        int    radius;
        double eb;
        FP     ebx2, ebx2_r, eb_r;
    } config;

    size_t m, mxm;

    struct {
        struct {
            size_t num_bits, num_uints, revbook_nbyte;
        } meta;
    } huffman;

    struct {
        uint8_t *host, *dev;
    } csr_file;
    OutlierHandler<Data>* csr;

    argpack*     ctx;
    cusz_header* header;
    BYTE*        header_byte;  // to use

    Decompressor(cusz_header* header, argpack* ctx);

    Decompressor(BYTE* in_dump);

    ~Decompressor() { delete csr; }

    Decompressor& huffman_decode(Capsule<Quant>* quant);

    Decompressor& scatter_outlier(Data* outlier);

    Decompressor& reversed_predict_quantize(Data* xdata, dim3 xyz, Data* anchor, Quant* quant);

    Decompressor& calculate_archive_nbyte();

    Decompressor& try_report_time();

    Decompressor& try_compare(Data* xdata);

    Decompressor& try_write2disk(Data* host_xdata);
};

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz_compress(argpack* ctx, Capsule<typename DataTrait<If_FP, DataByte>::Data>* in_data);

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz_decompress(argpack* ctx);

#endif
