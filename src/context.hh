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

#include "common.hh"
#include "utils/format.hh"

using std::string;

extern const char* VERSION_TEXT;
extern const int   version;
extern const int   compatibility;

class cuszCTX {
   public:
    // clang-format off
    // on-off's
    struct { bool construct{false}, reconstruct{false}, dryrun{false}, experiment{false}; bool gtest{false}; } task_is;

    struct { bool binning{false}, logtransform{false}, prescan{false}; } preprocess;
    struct { bool gpu_nvcomp_cascade{false}, cpu_gzip{false}; } postcompress;

    struct { bool use_demo{false}, use_anchor{false}, autotune_huffchunk{true}; } on_off;
    struct { bool write2disk{false}, huffman{false}; } to_skip;
    struct { bool book{false}, quant{false}; } export_raw;
    struct { bool quality{true}, time{false}, cr{false}, compressibility{false}, dataseg{false}; } report;

    // filenames
    struct { string path2file, origin_cmp, path_basename, basename; } fnames;
    // clang-format on

    bool verbose{false};

    stat_t stat;

    int read_args_status{0};

    string opath;

    string demo_dataset;
    string dtype     = ConfigHelper::get_default_dtype();      // "f32"
    string mode      = ConfigHelper::get_default_cuszmode();   // "r2r"
    string predictor = ConfigHelper::get_default_predictor();  // "lorenzo"
    string reducer   = ConfigHelper::get_default_reducer();    // "huffman"

    uint32_t quant_nbyte{2}, huff_nbyte{4};

    int nnz_outlier;

    size_t huffman_num_uints, huffman_num_bits;
    int    huffman_chunk{512}, nchunk{-1};

    size_t       data_len{1}, quant_len{1}, anchor_len{1};
    unsigned int x, y, z, w;
    int          ndim{-1};

    double eb{0.0};
    int    dict_size{1024}, radius{512};

    void load_demo_sizes();

   private:
    void sort_out_fnames();

    void trap(int _status);

    void check_args_when_cli();

    static void print_short_doc();

    static void print_full_doc();

   public:
    cuszCTX(int argc, char** argv);

    cuszCTX(const char*, bool to_compress, bool dbg_print = false);
};

#endif  // ARGPARSE_HH
