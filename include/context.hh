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

#include "common/definition.hh"
#include "utils/configs.hh"
#include "utils/format.hh"
#include "utils/strhelper.hh"

namespace cusz {

extern const char* VERSION_TEXT;
extern const int   version;
extern const int   compatibility;

}  // namespace cusz

struct cuszCTX {
   public:
    // on-off's
    struct {
        bool construct{false}, reconstruct{false}, dryrun{false};
        bool experiment{false};
        bool gtest{false};
    } cli_task;

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
        std::string fname, origin_cmp, path_basename, basename, compress_output;
    } fname;

    bool verbose{false};

    // Stat stat;

    int read_args_status{0};

    std::string opath;

    std::string demo_dataset;
    std::string dtype     = ConfigHelper::get_default_dtype();      // "f32"
    std::string mode      = ConfigHelper::get_default_cuszmode();   // "r2r"
    std::string predictor = ConfigHelper::get_default_predictor();  // "lorenzo"
    std::string codec     = ConfigHelper::get_default_codec();      // "huffman-coarse"
    std::string spcodec   = ConfigHelper::get_default_spcodec();    // "cusparse-csr"
    std::string pipeline  = "auto";

    // sparsity related: init_nnz when setting up Spcodec
    float nz_density{0.25};
    float nz_density_factor{4};

    uint32_t codecs_in_use{0b01};

    uint32_t quant_bytewidth{2}, huff_bytewidth{4};

    bool codec_force_fallback() const { return huff_bytewidth == 8; }

    size_t huffman_num_uints, huffman_num_bits;
    int    vle_sublen{512}, vle_pardeg{-1};

    unsigned int x{1}, y{1}, z{1}, w{1};

    struct {
        // size_t x, y, z, w;
        size_t len;
    } rtlen;

    size_t data_len{1}, quant_len{1}, anchor_len{1};
    int    ndim{-1};

    size_t get_len() const { return data_len; }

    double eb{0.0};
    int    dict_size{1024}, radius{512};

    void load_demo_sizes();

    /*******************************************************************************
     * another configuration method, alternative to
     *******************************************************************************/
   public:
    // for configuration
    cuszCTX& set_eb(double _)
    {
        eb = _;
        return *this;
    }

    cuszCTX& set_radius(int _)
    {
        radius    = _;
        dict_size = radius * 2;
        return *this;
    }

    cuszCTX& set_huffbyte(int _)
    {
        huff_bytewidth = _;
        codecs_in_use  = codec_force_fallback() ? 0b11 /*use both*/ : 0b01 /*use 4-byte*/;
        return *this;
    }

    cuszCTX& set_huffchunk(int _)
    {
        vle_sublen              = _;
        use.autotune_vle_pardeg = false;
        return *this;
    }

    cuszCTX& set_spcodec_densityfactor(int _)
    {
        if (_ <= 1)
            throw std::runtime_error(
                "Density factor for Spcodec must be >1. For example, setting the factor as 4 indicates the density "
                "(the portion of nonzeros) is 25% in an array.");
        nz_density_factor = _;
        nz_density        = 1.0 / _;
        return *this;
    }

    cuszCTX& enable_anchor(bool _)
    {
        use.anchor = true;
        return *this;
    }
    cuszCTX& enable_input_nondestructive(bool _)
    {
        // placeholder
        return *this;
    }

    cuszCTX& enable_failfast(bool _)
    {
        // placeholder
        return *this;
    }

    cuszCTX& set_alloclen(size_t _)
    {
        rtlen.len = _;
        return *this;
    }

    cuszCTX& set_control_string(const char* in_str);

    cuszCTX& use_anchor(size_t _)
    {
        use.anchor = true;
        return *this;
    }

    // set x, y, z, w, ndim, data_len
    cuszCTX& set_len(size_t _x, size_t _y = 1, size_t _z = 1, size_t _w = 1)
    {
        x = _x, y = _y, z = _z, w = _w;

        ndim = 4;
        if (w == 1) ndim = 3;
        if (z == 1) ndim = 2;
        if (y == 1) ndim = 1;

        data_len = x * y * z * w;

        if (data_len == 1) throw std::runtime_error("Input data length cannot be 1 (in 1-D view).");
        if (data_len == 0) throw std::runtime_error("Input data length cannot be 0 (in 1-D view).");

        return *this;
    }

   private:
    void derive_fnames();

    void validate();

   public:
    void trap(int _status);

    static void print_doc(bool full = false);

   public:
    static void parse_input_length(const char* lenstr, cuszCTX* ctx)
    {
        std::vector<std::string> dims;
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
    cuszCTX() = default;

    cuszCTX(int argc, char** argv);

    cuszCTX(const char*, bool dbg_print = false);
};

typedef struct cuszCTX cusz_context;

namespace cusz {

using Context   = cusz_context;
using context_t = cusz_context*;

}  // namespace cusz

#endif  // ARGPARSE_HH
