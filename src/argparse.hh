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
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cstdlib>
#include <iostream>
#include <regex>
#include <string>
#include "format.hh"

using std::string;

extern const char* version_text;
extern const int   version;
extern const int   compatibility;

typedef struct ArgPack {
    int read_args_status;

    string fname;
    string mode;  // abs (absolute), r2r (relative to value range)
    string demo_dataset;
    string alt_xout_name;
    string dtype;
    int    dict_size;
    int    quant_rep;
    int    input_rep;        // for standalone huffman
    int    huffman_datalen;  // for standalone huffman
    int    huffman_rep;
    int    huffman_chunk;
    int    n_dim;
    int    d0;
    int    d1;
    int    d2;
    int    d3;
    double mantissa;
    double exponent;
    bool   to_archive;
    bool   to_extract;
    bool   to_encode;    // for standalone huffman
    bool   to_decode;    // for standalone huffman
    bool   get_entropy;  // for standalone huffman (not in use)
    bool   use_demo;
    bool   verbose;
    bool   to_verify;
    bool   verify_huffman;
    bool   skip_huffman;
    bool   skip_writex;
    bool   pre_binning;
    bool   dry_run;

    static string format(const string& s);

    int trap(int _status);

    void CheckArgs();

    void HuffmanCheckArgs();

    static void cuszDoc();

    static void HuffmanDoc();

    static void cuszFullDoc();

    ArgPack(int argc, char** argv);

    ArgPack(int argc, char** argv, bool standalone_huffman);

} argpack;

#endif  // ARGPARSE_HH
