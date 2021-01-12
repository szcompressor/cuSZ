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

typedef struct ArgPack {
    // TODO [ ] metadata
    // TODO [x] metric/stat
    stat_t stat;

    int read_args_status{0};

    string cx_path2file;
    string c_huff_base, c_fo_q, c_fo_outlier, c_fo_yamp;
    string x_fi_q, x_fi_outlier, x_fi_yamp, x_fo_xd;
    string x_fi_origin;

    string mode;  // abs (absolute), r2r (relative to value range)
    string demo_dataset;
    string opath;
    string dtype;
    int    dict_size{1024};
    int    quant_byte{2}, huff_byte{4};
    int    huffman_chunk{512};
    int    n_dim{-1}, d0{1}, d1{1}, d2{1}, d3{1};
    double mantissa{1.0}, exponent{-4.0};
    bool   to_archive{false}, to_extract{false}, to_dryrun{false};
    bool   autotune_huffman_chunk{true};
    bool   use_demo{false};
    bool   verbose{false}, to_verify{false};
    bool   export_codebook{false};
    bool   verify_huffman{false};
    bool   skip_huffman{false}, skip_writex{false};
    bool   pre_binning{false};

    int  input_rep;        // for standalone huffman
    int  huffman_datalen;  // for standalone huffman
    bool to_encode;        // for standalone huffman
    bool to_decode;        // for standalone huffman
    bool get_entropy;      // for standalone huffman (not in use)
    bool to_gzip;          // wenyu: whether to do a gzip lossless compression on encoded data

    static string format(const string& s);

    int trap(int _status);

    void CheckArgs();

    void HuffmanCheckArgs();

    static void cuszDoc();

    static void HuffmanDoc();

    static void cuszFullDoc();

    ArgPack(int argc, char** argv);

    ArgPack(int argc, char** argv, bool standalone_huffman);

    void SortOutFilenames();

} argpack;

#endif  // ARGPARSE_HH
