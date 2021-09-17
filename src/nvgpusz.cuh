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

    void report_compression_time();

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
    cuszCTX* ctx;
    //
    cusz_header* header;

    // OOD, indicated by v2
    cusz_header header_v2;
    dim3        xyz_v2;

    unsigned int get_revbook_nbyte() { return sizeof(Huff) * (2 * TYPE_BITCOUNT) + sizeof(Quant) * length.dict_size; }

    Compressor(cuszCTX* _ctx);

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

    void compress(Capsule<Data>* in_data);

    void consolidate(bool on_cpu = true, bool on_gpu = false);

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

    void read_array_nbyte_from_header();

    void get_data_seg_offsets();

   public:
    void register_spline3(Spline3<Data*, Quant*, float>* _spline3) { spline3 = _spline3; }

    Spline3<Data*, Quant*, FP>* spline3;

    dim3 xyz;

    void decompress();

    struct {
        BYTE* whole;
    } consolidated_dump;

    std::vector<uint32_t> offsets;

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

    cuszCTX*     ctx;
    cusz_header* header;
    BYTE*        header_byte;  // to use

    Decompressor(cusz_header* header, cuszCTX* ctx);

    Decompressor(cuszCTX* ctx);

    Decompressor(cuszCTX* ctx, uint8_t* in_dump);

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
void cusz_compress(cuszCTX* ctx, Capsule<typename DataTrait<If_FP, DataByte>::Data>* in_data);

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz_decompress(cuszCTX* ctx);

#endif
