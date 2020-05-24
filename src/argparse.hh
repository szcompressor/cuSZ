//
// Created by jtian on 4/24/20.
//

#ifndef ARGPARSE_HH
#define ARGPARSE_HH

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

    static void PrintInstruction();

    static void PrintHelpDoc();

    ArgPack(int argc, char** argv);

} argpack;

#endif  // ARGPARSE_HH
