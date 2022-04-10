#ifndef ARGPARSE_HH
#define ARGPARSE_HH

/**
 * @file argparse.hh
 * @author Jiannan Tian
 * @brief Argument parser (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on: 20-04-24
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cstdlib>
#include <iostream>
#include <regex>
#include <string>

#include "common/configs.hh"
#include "common/definition.hh"
#include "utils/format.hh"
#include "utils/strhelper.hh"

using std::string;

namespace cusz {

extern const char* VERSION_TEXT;
extern const int   version;
extern const int   compatibility;

}  // namespace cusz

class cuszCTX {
   public:
    // on-off's
    struct {
        bool construct{false}, reconstruct{false}, dryrun{false};
        bool experiment{false};
        bool gtest{false};
    } task_is;

    struct {
        bool binning{false}, logtransform{false}, prescan{false};
    } preprocess;
    struct {
        bool gpu_nvcomp_cascade{false}, cpu_gzip{false};
    } postcompress;

    struct {
        bool predefined_demo{false}, release_input{false};
        bool anchor{false}, autotune_vle_pardeg{true}, gpu_verify{false};
    } use;

    struct {
        bool book{false}, quant{false};
    } export_raw;

    struct {
        bool write2disk{false}, huffman{false};
    } skip;
    struct {
        bool time{false}, cr{false}, compressibility{false};
    } report;

    // filenames
    struct {
        string fname, origin_cmp, path_basename, basename, compress_output;
    } fname;

    bool verbose{false};

    // Stat stat;

    int read_args_status{0};

    string opath;

    string demo_dataset;
    string dtype{ConfigHelper::get_default_dtype()};          // "f32"
    string mode{ConfigHelper::get_default_cuszmode()};        // "r2r"
    string predictor{ConfigHelper::get_default_predictor()};  // "lorenzo"
    string codec{ConfigHelper::get_default_codec()};          // "huffman-coarse"
    string spcodec{ConfigHelper::get_default_spcodec()};      // "cusparse-csr"
    string compression_pipeline{"auto"};

    // sparsity related: init_nnz when setting up SpCodec
    float nz_density{SparseMethodSetup::default_density};
    float nz_density_factor{SparseMethodSetup::default_density_factor};

    uint32_t codecs_in_use{0b01};

    uint32_t quant_bytewidth{2}, huff_bytewidth{4};

    bool codec_force_fallback() const { return huff_bytewidth == 8; }

    size_t huffman_num_uints, huffman_num_bits;
    int    vle_sublen{512}, vle_pardeg{-1};

    unsigned int x, y, z, w;
    size_t       data_len{1}, quant_len{1}, anchor_len{1};
    int          ndim{-1};

    size_t get_len() const { return data_len; }

    double eb{0.0};
    int    dict_size{1024}, radius{512};

    void load_demo_sizes();

   private:
    void derive_fnames();

    void check_cli_args();

   public:
    void trap(int _status);

    static void print_short_doc();

    static void print_full_doc();

    static int autotune(cuszCTX* ctx);

   public:
    static void parse_input_length(const char* lenstr, cuszCTX* ctx)
    {
        std::vector<string> dims;
        ConfigHelper::parse_length_literal(lenstr, dims);
        ctx->ndim = dims.size();
        ctx->y = ctx->z = ctx->w = 1;
        ctx->x                   = StrHelper::str2int(dims[0]);
        if (ctx->ndim >= 2) ctx->y = StrHelper::str2int(dims[1]);
        if (ctx->ndim >= 3) ctx->z = StrHelper::str2int(dims[2]);
        if (ctx->ndim >= 4) ctx->w = StrHelper::str2int(dims[3]);
        ctx->data_len = ctx->x * ctx->y * ctx->z * ctx->w;
    }

   public:
    cuszCTX(int argc, char** argv);

    cuszCTX(const char*, bool dbg_print = false);
};

using cuszCONFIG = cuszCTX;

namespace cusz {

using Context   = cuszCTX;
using context_t = Context*;

}  // namespace cusz

#endif  // ARGPARSE_HH
