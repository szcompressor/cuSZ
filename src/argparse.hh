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

#include "types.hh"
#include "utils/format.hh"

using std::string;

extern const char* version_text;
extern const int   version;
extern const int   compatibility;

class ArgPack {
   private:
   public:
    void load_demo_sizes();
    struct {
        bool   use_demo_dataset{false};
        bool   pre_binning{false}, pre_log_trans{false};
        bool   autotune_huffchunk{true};
        bool   construct{false}, reconstruct{false}, dryrun{false};
        bool   export_book{false}, export_quant{false};
        bool   skip_write2disk{false}, skip_huffman{false};
        bool   lossless_nvcomp_cascade{false}, lossless_gzip{false};
        bool   gtest{false};
        string predictor = string("lorenzo");
    } task_is;

    struct {
        bool quality{true}, time{false}, cr{false};
        bool compressibility{false};
    } report;

    struct {
        string path2file;
        string origin_cmp;
        string path_basename;
        string basename;
    } fnames;

    stat_t stat;

    int read_args_status{0};

    string mode;  // abs (absolute), r2r (relative to value range)
    string demo_dataset;
    string opath;
    string dtype;

    int quant_nbyte{2}, huff_nbyte{4};
    int huffman_chunk{512};
    int nchunk{-1};
    int ndim{-1};

    double eb{0.0};

    size_t       data_len{1};
    unsigned int x, y, z, w;

    int dict_size{1024}, radius{512};

    bool verbose{false};

    // for standalone Huffman
    int input_rep{2};
    int huffman_datalen{-1};

    static string format(const string& s);

    int trap(int _status);

    void check_args();

    static void print_cusz_short_doc();

    static void print_cusz_full_doc();

    ArgPack() = default;

    size_t self_multiply4() { return x * y * z * w; };

    void parse_args(int argc, char** argv);

    void sort_out_fnames();
};

typedef ArgPack argpack;

#endif  // ARGPARSE_HH
