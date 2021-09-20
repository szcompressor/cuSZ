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

#include "type_trait.hh"
#include "types.hh"
#include "utils/format.hh"

using std::string;

extern const char* VERSION_TEXT;
extern const int   version;
extern const int   compatibility;

class cuszCTX {
   public:
    struct {
        bool   use_demo_dataset{false};
        bool   pre_binning{false}, pre_log_trans{false};
        bool   autotune_huffchunk{true};
        bool   construct{false}, reconstruct{false}, dryrun{false}, experiment{false};
        bool   export_book{false}, export_quant{false};
        bool   skip_write2disk{false}, skip_huffman{false};
        bool   lossless_nvcomp_cascade{false}, lossless_gzip{false};
        bool   gtest{false};
        bool   use_anchor{false};
        string predictor = string("lorenzo");
    } task_is;

    bool verbose{false};

    struct {
        bool quality{true}, time{false}, cr{false}, compressibility{false}, dataseg{false};
    } report;

    struct {
        string path2file, origin_cmp, path_basename, basename;
    } fnames;

    stat_t stat;

    int read_args_status{0};

    string mode;  // abs (absolute), r2r (relative to value range)
    string demo_dataset;
    string opath;
    string dtype;

    uint32_t quant_nbyte{2}, huff_nbyte{4};

    int nnz_outlier;

    size_t huffman_num_uints, huffman_num_bits;
    int    huffman_chunk{512};
    int    nchunk{-1};

    size_t       data_len{1}, quant_len{1}, anchor_len{1};
    unsigned int x, y, z, w;
    int          ndim{-1};

    double eb{0.0};
    int    dict_size{1024}, radius{512};

    void load_demo_sizes();

   private:
    void sort_out_fnames();

    void trap(int _status);

    void check_args();

    static void print_short_doc();

    static void print_full_doc();

   public:
    cuszCTX(int argc, char** argv);
};

#endif  // ARGPARSE_HH
