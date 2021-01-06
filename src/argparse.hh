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

    int read_args_status;

    string cx_path2file;
    string c_huff_base, c_fo_q, c_fo_outlier, c_fo_yamp;
    string x_fi_q, x_fi_outlier, x_fi_yamp, x_fo_xd;
    string x_fi_origin;

    string mode;  // abs (absolute), r2r (relative to value range)
    string demo_dataset;
    string opath;
    string dtype;
    int    dict_size;
    int    quant_byte;
    int    huff_byte;
    int    huffman_chunk;
    int    n_dim, d0, d1, d2, d3;
    double mantissa, exponent;
    bool   to_archive, to_extract, to_dryrun;
    bool   autotune_huffman_chunk;
    bool   use_demo;
    bool   verbose;
    bool   to_verify;
    bool   verify_huffman;
    bool   skip_huffman, skip_writex;
    bool   pre_binning;

    int  input_rep;        // for standalone huffman
    int  huffman_datalen;  // for standalone huffman
    bool to_encode;        // for standalone huffman
    bool to_decode;        // for standalone huffman
    bool get_entropy;      // for standalone huffman (not in use)
    bool to_gzip;          // wenyu: whether to do a gzip lossless compression on encoded data
    bool to_nvcomp;        // whether or not to activate nvidia parallel cascading compression
    bool to_gtest;         // whether or not to activate unit test

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
