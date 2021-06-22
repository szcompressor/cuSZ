/**
 * @file argparse.cc
 * @author Jiannan Tian
 * @brief Argument parser.
 * @version 0.1
 * @date 2020-09-20
 * Created on: 20-04-24
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include "argparse.hh"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <regex>
#include <string>
#include "argument_parser/document.hh"
#include "metadata.hh"
#include "utils/format.hh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

// TODO update with default values

// TODO check version
const char* version_text  = "version: pre-alpha, build: 2020-09-20";
const int   version       = 200920;
const int   compatibility = 0;

string  //
ArgPack::format(const string& s)
{
    std::regex  bful("@(.*?)@");
    std::string bful_text("\e[1m\e[4m$1\e[0m");
    std::regex  bf("\\*(.*?)\\*");
    std::string bf_text("\e[1m$1\e[0m");
    std::regex  ul(R"(_((\w|-|\d|\.)+?)_)");
    std::string ul_text("\e[4m$1\e[0m");
    std::regex  red(R"(\^\^(.*?)\^\^)");
    std::string red_text("\e[31m$1\e[0m");
    auto        a = std::regex_replace(s, bful, bful_text);
    auto        b = std::regex_replace(a, bf, bf_text);
    auto        c = std::regex_replace(b, ul, ul_text);
    auto        d = std::regex_replace(c, red, red_text);
    return d;
}

int  //
ArgPack::trap(int _status)
{
    this->read_args_status = _status;
    return read_args_status;
}

void  //
ArgPack::HuffmanCheckArgs()
{
    bool to_abort = false;
    if (subfiles.path2file.empty()) {
        cerr << log_err << "Not specifying input file!" << endl;
        to_abort = true;
    }
    if (SelfMultiple4(dim4) == 1 and not szwf.input_use_demo) {
        cerr << log_err << "Wrong input size(s)!" << endl;
        to_abort = true;
    }
    if (not hwf.encode and not hwf.decode and not hwf.dryrun) {
        cerr << log_err << "Select encode (-a), decode (-x) or dry-run (-r)!" << endl;
        to_abort = true;
    }
    // if (dtype != "f32" and dtype != "f64") {
    //     cout << dtype << endl;
    //     cerr << log_err << "Not specifying data type!" << endl;
    //     to_abort = true;
    // }

    if (input_rep == 1) {  // TODO
        assert(dict_size <= 256);
    }
    else if (input_rep == 2) {
        assert(dict_size <= 65536);
    }

    if (hwf.dryrun and hwf.encode and hwf.decode) {
        cerr << log_warn << "No need to dry-run, encode and decode at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        hwf.encode = false;
        hwf.decode = false;
    }
    else if (hwf.dryrun and hwf.encode) {
        cerr << log_warn << "No need to dry-run and encode at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        hwf.encode = false;
    }
    else if (hwf.dryrun and hwf.decode) {
        cerr << log_warn << "No need to dry-run and decode at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        hwf.decode = false;
    }

    if (to_abort) {
        cuszDoc();
        exit(-1);
    }
}

void  //
ArgPack::CheckArgs()
{
    bool to_abort = false;
    if (subfiles.path2file.empty()) {
        cerr << log_err << "Not specifying input file!" << endl;
        to_abort = true;
    }
    if (SelfMultiple4(dim4) == 1 and not szwf.input_use_demo) {
        if (szwf.lossy_construct or szwf.lossy_dryrun) {
            cerr << log_err << "Wrong input size(s)!" << endl;
            to_abort = true;
        }
    }
    if (not szwf.lossy_construct and not szwf.lossy_reconstruct and not szwf.lossy_dryrun) {
        cerr << log_err << "Select compress (-a), decompress (-x) or dry-run (-r)!" << endl;
        to_abort = true;
    }
    if (dtype != "f32" and dtype != "f64") {
        if (szwf.lossy_construct or szwf.lossy_dryrun) {
            cout << dtype << endl;
            cerr << log_err << "Not specifying data type!" << endl;
            to_abort = true;
        }
    }

    if (quant_byte == 1) {  // TODO
        assert(dict_size <= 256);
    }
    else if (quant_byte == 2) {
        assert(dict_size <= 65536);
    }

    if (szwf.lossy_dryrun and szwf.lossy_construct and szwf.lossy_reconstruct) {
        cerr << log_warn << "No need to dry-run, compress and decompress at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        szwf.lossy_construct   = false;
        szwf.lossy_reconstruct = false;
    }
    else if (szwf.lossy_dryrun and szwf.lossy_construct) {
        cerr << log_warn << "No need to dry-run and compress at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        szwf.lossy_construct = false;
    }
    else if (szwf.lossy_dryrun and szwf.lossy_reconstruct) {
        cerr << log_warn << "No need to dry-run and decompress at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        szwf.lossy_reconstruct = false;
    }

    // if (szwf.gtest) {
    //     if (szwf.lossy_dryrun) { szwf.gtest = false; }
    //     else {
    //         if (not(szwf.lossy_construct and szwf.lossy_reconstruct)) { szwf.gtest = false; }
    //         if (subfiles.decompress.in_origin == "") { szwf.gtest = false; }
    //     }
    // }

    if (to_abort) {
        cuszDoc();
        exit(-1);
    }
}

void  //
ArgPack::HuffmanDoc()
{
    const string instruction =
        "\n"
        "OVERVIEW: Huffman submodule as standalone program\n"  // TODO from this line on
        "\n"
        "USAGE:\n"
        "  The basic use with demo datum is listed below,\n"
        "    ./huff --encode --decode --verify --input ./baryon_density.dat.b16 \\\n"
        "        -3 512 512 512 --input-rep 16 --huffman-rep 32 --huffman-chunk 2048 --dict-size 1024\n"
        "  or shorter\n"
        "    ./huff -e -d -V -i ./baryon_density.dat.b16 -3 512 512 512 -R 16 -H 32 -C 2048 -c 1024\n"
        "            ^  ^  ^ ~~~~~~~~~~~~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~ ~~~~~ ~~~~~ ~~~~~~~ ~~~~~~~\n"
        "            |  |  |       input datum file         dimension   input Huff. Huff.   codebook\n"
        "          enc dec verify                                       rep.  rep.  chunk   size\n"
        "\n"
        "EXAMPLES\n"
        "  Essential:\n"
        "    ./bin/huff -e -d -i ./baryon_density.dat.b16 -3 512 512 512 -R 16 -c 1024\n"
        "    have to input dimension, and higher dimension for a multiplication of each dim.,\n"
        "    as default values input-rep=16 (bits), huff-rep=32 (bits), codebokk-size=1024 (symbols)\n"
        "\n";
    cout << instruction << endl;
}

void  //
ArgPack::cuszDoc()
{
    cout << cusz_short_doc << endl;
}

void  //
ArgPack::cuszFullDoc()
{
    cout << format(cusz_full_doc) << endl;
}

void ArgPack::ParseHuffmanArgs(int argc, char** argv)
{
    if (argc == 1) {
        HuffmanDoc();
        exit(0);
    }

    auto str2int = [&](const char* s) {
        char* end;
        auto  res = std::strtol(s, &end, 10);
        if (*end) {
            const char* notif = "invalid option value, non-convertible part: ";
            cerr << log_err << notif << "\e[1m" << end << "\e[0m" << endl;
            cerr << string(log_null.length() + strlen(notif), ' ') << "\e[1m"  //
                 << string(strlen(end), '~')                                   //
                 << "\e[0m" << endl;
            trap(-1);
            return 0;  // just a placeholder
        }
        return (int)res;
    };

    auto str2fp = [&](const char* s) {
        char* end;
        auto  res = std::strtod(s, &end);
        if (*end) {
            const char* notif = "invalid option value, non-convertible part: ";
            cerr << log_err << notif << "\e[1m" << end << "\e[0m" << endl;
            cerr << string(log_null.length() + strlen(notif), ' ') << "\e[1m"  //
                 << string(strlen(end), '~')                                   //
                 << "\e[0m" << endl;
            trap(-1);
            return 0;  // just a placeholder
        }
        return (int)res;
    };

    int i = 1;
    while (i < argc) {
        if (argv[i][0] == '-') {
            switch (argv[i][1]) {
                // more readable args
                // ----------------------------------------------------------------
                case '-':
                    if (string(argv[i]) == "--help") goto tag_help;
                    // major features
                    if (string(argv[i]) == "--enc" or string(argv[i]) == "--encode") goto tag_encode;
                    if (string(argv[i]) == "--dec" or string(argv[i]) == "--decode") goto tag_decode;
                    if (string(argv[i]) == "--dry-run") goto tag_dryrun;
                    if (string(argv[i]) == "--verify") goto tag_verify;
                    if (string(argv[i]) == "--input") goto tag_input;
                    if (string(argv[i]) == "--entropy") goto tag_entropy;
                    if (string(argv[i]) == "--input-rep" or string(argv[i]) == "--interpret") goto tag_rep;
                    if (string(argv[i]) == "--huffman-rep") goto tag_huff_byte;
                    if (string(argv[i]) == "--huffman-chunk") goto tag_huff_chunk;
                    if (string(argv[i]) == "--dict-size") goto tag_dict;
                // work
                // ----------------------------------------------------------------
                case 'e':
                tag_encode:
                    hwf.encode = true;
                    break;
                case 'd':
                tag_decode:
                    hwf.decode = true;
                    break;
                case 'V':
                tag_verify:
                    cout << "fix 'verify-huffman' later" << endl;
                    // verify_huffman = true;  // TODO verify huffman in workflow
                    break;
                case 'r':
                tag_dryrun:
                    hwf.dryrun = true;
                    break;
                case 'E':
                tag_entropy:
                    get_huff_entropy = true;
                    break;
                // input dimensionality
                // ----------------------------------------------------------------
                case '1':
                case '2':
                case '3':
                case '4': cout << "fix later" << endl; break;
                // help document
                // ----------------------------------------------------------------
                case 'h':
                tag_help:
                    HuffmanDoc();
                    exit(0);
                    break;
                // input datum file
                // ----------------------------------------------------------------
                case 'i':
                tag_input:
                    if (i + 1 <= argc) { subfiles.path2file = string(argv[++i]); }
                    break;
                case 'R':
                tag_rep:
                    if (i + 1 <= argc) input_rep = str2int(argv[++i]);
                    break;
                case 'H':
                tag_huff_byte:
                    if (i + 1 <= argc) huff_byte = str2int(argv[++i]);
                    break;
                case 'C':
                tag_huff_chunk:
                    if (i + 1 <= argc) huffman_chunk = str2int(argv[++i]);
                    break;
                case 'c':
                tag_dict:
                    if (i + 1 <= argc) dict_size = str2int(argv[++i]);
                    break;
                default:
                    const char* notif_prefix = "invalid option at position ";
                    char*       notif;
                    int         size = asprintf(&notif, "%d: %s", i, argv[i]);
                    cerr << log_err << notif_prefix << "\e[1m" << notif << "\e[0m"
                         << "\n";
                    cerr << string(log_null.length() + strlen(notif_prefix), ' ');
                    cerr << "\e[1m";
                    cerr << string(strlen(notif), '~');
                    cerr << "\e[0m\n";
                    trap(-1);
            }
        }
        else {
            const char* notif_prefix = "invalid argument at position ";
            char*       notif;
            int         size = asprintf(&notif, "%d: %s", i, argv[i]);
            cerr << log_err << notif_prefix << "\e[1m" << notif
                 << "\e[0m"
                    "\n"
                 << string(log_null.length() + strlen(notif_prefix), ' ')  //
                 << "\e[1m"                                                //
                 << string(strlen(notif), '~')                             //
                 << "\e[0m\n";
            trap(-1);
        }
        i++;
    }

    // phase 1: check grammar
    if (read_args_status != 0) {
        cout << log_info << "Exiting..." << endl;
        // after printing ALL argument errors
        exit(-1);
    }

    // phase 2: check if meaningful
    HuffmanCheckArgs();
}

void ArgPack::ParseCuszArgs(int argc, char** argv)
{
    if (argc == 1) {
        cuszDoc();
        exit(0);
    }

    opath = "";

    auto str2int = [&](const char* s) {
        char* end;
        auto  res = std::strtol(s, &end, 10);
        if (*end) {
            const char* notif = "invalid option value, non-convertible part: ";
            cerr << log_err << notif << "\e[1m" << end << "\e[0m" << endl;
            cerr << string(log_null.length() + strlen(notif), ' ') << "\e[1m"  //
                 << string(strlen(end), '~')                                   //
                 << "\e[0m" << endl;
            trap(-1);
            return 0U;  // just a placeholder
        }
        return (unsigned int)res;
    };

    auto str2fp = [&](const char* s) {
        char* end;
        auto  res = std::strtod(s, &end);
        if (*end) {
            const char* notif = "invalid option value, non-convertible part: ";
            cerr << log_err << notif << "\e[1m" << end << "\e[0m" << endl;
            cerr << string(log_null.length() + strlen(notif), ' ') << "\e[1m"  //
                 << string(strlen(end), '~')                                   //
                 << "\e[0m" << endl;
            trap(-1);
            return 0;  // just a placeholder
        }
        return (int)res;
    };

    int i = 1;
    while (i < argc) {
        if (argv[i][0] == '-') {
            auto long_opt = string(argv[i]);
            switch (argv[i][1]) {
                // ----------------------------------------------------------------
                case '-':
                    if (long_opt == "--help") goto tag_help;              // DOCUMENT
                    if (long_opt == "--version") goto tag_version;        //
                    if (long_opt == "--verbose") goto tag_verbose;        //
                    if (long_opt == "--meta") goto tag_meta;              //
                    if (long_opt == "--mode") goto tag_mode;              // COMPRESSION CONFIG
                    if (long_opt == "--quant-byte") goto tag_quant_byte;  //
                    if (long_opt == "--huff-byte") goto tag_huff_byte;    //
                    if (long_opt == "--huff-chunk") goto tag_huff_chunk;  //
                    if (long_opt == "--eb") goto tag_error_bound;         //
                    if (long_opt == "--dict-size") goto tag_dict;         //
                    if (long_opt == "--dtype") goto tag_type;             //
                    if (long_opt == "--input") goto tag_input;            // INPUT
                    if (long_opt == "--demo") goto tag_demo;              //
                    if (long_opt == "--verify") goto tag_verify;          //
                    if (long_opt == "--len") goto tag_len;                //
                    if (long_opt == "--part") goto tag_partition;         //
                    if (long_opt == "--compress") goto tag_compress;      // WORKFLOW
                    if (long_opt == "--zip") goto tag_compress;           //
                    if (long_opt == "--decompress") goto tag_decompress;  //
                    if (long_opt == "--unzip") goto tag_decompress;       //
                    if (long_opt == "--dry-run") goto tag_dryrun;         //
                    if (long_opt == "--skip") goto tag_excl;              //
                    if (long_opt == "--exclude") goto tag_excl;           //
                    if (long_opt == "--pre") goto tag_preproc;            // IO
                    if (long_opt == "--analysis") goto tag_analysis;      //
                    if (long_opt == "--output") goto tag_x_out;           //
                    if (long_opt == "--partition-experiment") { szwf.exp_partitioning_imbalance = true; }
                    if (long_opt == "--opath") {  // TODO the followings has no single-letter options
                        if (i + 1 <= argc)
                            this->opath = string(argv[++i]);  // TODO does not apply for preprocessed such as binning
                        break;
                    }
                    if (long_opt == "--origin") {
                        if (i + 1 <= argc) subfiles.decompress.in_origin = string(argv[++i]);
                        break;
                    }
                    if (long_opt == "--gzip") {
                        szwf.lossless_gzip = true;
                        break;  // wenyu: if there is "--gzip", set member field to_gzip true
                    }
                    if (long_opt == "--nvcomp") {
                        throw std::runtime_error(
                            "[argparse] nvcomp is disabled temporarily in favor of code refactoring.");
                        szwf.lossless_nvcomp_cascade = false;
                        break;
                    }
                    if (long_opt == "--gtest") {
                        throw std::runtime_error(
                            "[argparse] gtest is disabled temporarily in favor of code refactoring.");
                        szwf.gtest = false;
                        break;
                    }
                    // if (long_opt == "--coname") {
                    //     // TODO does not apply for preprocessed such as binning
                    //     if (i + 1 <= argc) ap->coname = string(argv[++i]);
                    //     break;
                    // }
                    // if (long_opt == "--xoname") {
                    //     // TODO does not apply for preprocessed such as binning
                    //     if (i + 1 <= argc) ap->xoname = string(argv[++i]);
                    //     break;
                    // }
                // WORKFLOW
                case 'z':
                tag_compress:
                    szwf.lossy_construct = true;
                    break;
                case 'x':
                tag_decompress:
                    szwf.lossy_reconstruct = true;
                    break;
                case 'r':
                tag_dryrun:
                    szwf.lossy_dryrun = true;
                    break;
                // COMPRESSION CONFIG
                case 'm':  // mode
                tag_mode:
                    if (i + 1 <= argc) mode = string(argv[++i]);
                    break;
                // OTHER WORKFLOW
                case 'A':
                tag_analysis:
                    if (i + 1 <= argc) {
                        string exclude(argv[++i]);
                        if (exclude.find("export-codebook") != std::string::npos) { szwf.exp_export_codebook = true; }
                        if (exclude.find("huff-entropy") != std::string::npos) get_huff_entropy = true;
                        if (exclude.find("huff-avg-bitcount") != std::string::npos) get_huff_avg_bitcount = true;
                    }
                    break;
                case 'S':
                tag_excl:
                    if (i + 1 <= argc) {
                        string exclude(argv[++i]);
                        if (exclude.find("huffman") != std::string::npos) { szwf.skip_huffman_enc = true; }
                        if (exclude.find("write.x") != std::string::npos) { szwf.skip_write_output = true; }
                    }
                    break;
                // INPUT
                case 'l':
                tag_len:
                    if (i + 1 <= argc) {
                        std::stringstream   datalen(argv[++i]);
                        std::vector<string> dims;
                        while (datalen.good()) {
                            string substr;
                            getline(datalen, substr, ',');
                            dims.push_back(substr);
                        }
                        ndim = dims.size();
                        if (ndim == 1) {  //
                            auto d0 = str2int(dims[0].c_str());
                            dim4    = {d0, 1, 1, 1};
                        }
                        if (ndim == 2) {  //
                            auto d0 = str2int(dims[0].c_str()), d1 = str2int(dims[1].c_str());
                            dim4 = {d0, d1, 1, 1};
                        }
                        if (ndim == 3) {
                            auto d0 = str2int(dims[0].c_str()), d1 = str2int(dims[1].c_str());
                            auto d2 = str2int(dims[2].c_str());
                            dim4    = {d0, d1, d2, 1};
                        }
                        if (ndim == 4) {
                            auto d0 = str2int(dims[0].c_str()), d1 = str2int(dims[1].c_str());
                            auto d2 = str2int(dims[2].c_str()), d3 = str2int(dims[3].c_str());
                            dim4 = {d0, d1, d2, d3};
                        }
                    }
                    break;
                case 'p':
                tag_partition:
                    if (i + 1 <= argc) {
                        std::stringstream   datalen(argv[++i]);
                        std::vector<string> parts;
                        while (datalen.good()) {
                            string substr;
                            getline(datalen, substr, ',');
                            parts.push_back(substr);
                        }
                        ndim = parts.size();
                        if (ndim == 1) {  //
                            auto p0 = str2int(parts[0].c_str());
                            part4   = {p0, 1, 1, 1};
                        }
                        if (ndim == 2) {  //
                            auto p0 = str2int(parts[0].c_str()), p1 = str2int(parts[1].c_str());
                            part4 = {p0, p1, 1, 1};
                        }
                        if (ndim == 3) {
                            auto p0 = str2int(parts[0].c_str()), p1 = str2int(parts[1].c_str());
                            auto p2 = str2int(parts[2].c_str());
                            part4   = {p0, p1, p2, 1};
                        }
                        if (ndim == 4) {
                            auto p0 = str2int(parts[0].c_str()), p1 = str2int(parts[1].c_str());
                            auto p2 = str2int(parts[2].c_str()), p3 = str2int(parts[3].c_str());
                            part4 = {p0, p1, p2, p3};
                        }
                    }
                    break;
                case 'i':
                tag_input:
                    if (i + 1 <= argc) subfiles.path2file = string(argv[++i]);
                    break;
                    // alternative output
                case 'o':
                tag_x_out:
                    cerr << log_err
                         << "\"-o\" will be working in the (near) future release. Pleae use \"--opath [path]\" "
                            "to "
                            "specify output path."
                         << endl;
                    exit(1);
                    break;
                // preprocess
                case 'P':
                tag_preproc:
                    if (i + 1 <= argc) {
                        string pre(argv[++i]);
                        if (pre.find("binning") != std::string::npos) { szwf.pre_binning = true; }
                    }
                    break;
                // demo datasets
                case 'D':
                tag_demo:
                    if (i + 1 <= argc) {
                        szwf.input_use_demo = true;
                        demo_dataset        = string(argv[++i]);
                    }
                    break;
                // DOCUMENT
                case 'h':
                tag_help:
                    cuszFullDoc();
                    exit(0);
                    break;
                case 'v':
                tag_version:
                    // TODO
                    cout << log_info << version_text << endl;
                    break;
                // COMPRESSION CONFIG
                case 't':
                tag_type:
                    if (i + 1 <= argc) {
                        string s = string(string(argv[++i]));
                        if (s == "f32" or s == "fp4")
                            dtype = "f32";
                        else if (s == "f64" or s == "fp8")
                            dtype = "f64";
                        // if (string(argv[++i]) == "i16") dtype = "i16";
                        // if (string(argv[++i]) == "i32") dtype = "i32";
                        // if (string(argv[++i]) == "i64") dtype = "i64";
                    }
                    break;
                case 'M':
                tag_meta:  // TODO print .sz archive metadata
                    break;
                // internal representation and size
                case 'Q':
                tag_quant_byte:
                    if (i + 1 <= argc) quant_byte = str2int(argv[++i]);
                    break;
                case 'H':
                tag_huff_byte:
                    if (i + 1 <= argc) huff_byte = str2int(argv[++i]);
                    break;
                case 'C':
                tag_huff_chunk:
                    if (i + 1 <= argc) {  //
                        huffman_chunk               = str2int(argv[++i]);
                        szwf.autotune_huffman_chunk = false;
                    }
                    break;
                case 'e':
                tag_error_bound:
                    if (i + 1 <= argc) {
                        char* end;
                        this->eb = std::strtod(argv[++i], &end);
                    }
                    break;
                case 'y':
                tag_verify:
                    if (i + 1 <= argc) {
                        string veri(argv[++i]);
                        if (veri.find("huffman") != std::string::npos) { szwf.verify_huffman = true; }
                        // TODO verify data quality
                    }
                    break;
                case 'V':
                tag_verbose:
                    verbose = true;
                    break;
                case 'd':
                tag_dict:
                    if (i + 1 <= argc) {
                        dict_size = str2int(argv[++i]);
                        radius    = dict_size / 2;
                    }
                    break;
                default:
                    const char* notif_prefix = "invalid option value at position ";
                    char*       notif;
                    int         size = asprintf(&notif, "%d: %s", i, argv[i]);
                    cerr << log_err << notif_prefix << "\e[1m" << notif << "\e[0m"
                         << "\n";
                    cerr << string(log_null.length() + strlen(notif_prefix), ' ');
                    cerr << "\e[1m";
                    cerr << string(strlen(notif), '~');
                    cerr << "\e[0m\n";
                    trap(-1);
            }
        }
        else {
            const char* notif_prefix = "invalid option at position ";
            char*       notif;
            int         size = asprintf(&notif, "%d: %s", i, argv[i]);
            cerr << log_err << notif_prefix << "\e[1m" << notif
                 << "\e[0m"
                    "\n"
                 << string(log_null.length() + strlen(notif_prefix), ' ')  //
                 << "\e[1m"                                                //
                 << string(strlen(notif), '~')                             //
                 << "\e[0m\n";
            trap(-1);
        }
        i++;
    }

    // phase 1: check grammar
    if (read_args_status != 0) {
        cout << log_info << "Exiting..." << endl;
        // after printing ALL argument errors
        exit(-1);
    }

    // phase 2: check if meaningful
    CheckArgs();
    // phase 3: sort out filenames
    SortOutFilenames();
}

void ArgPack::SortOutFilenames()
{
    // (1) "fname"          -> "", "fname"
    // (2) "./fname"        -> "./" "fname"
    // (3) "/path/to/fname" -> "/path/to", "fname"
    auto input_path = subfiles.path2file.substr(0, subfiles.path2file.rfind('/') + 1);
    if (not szwf.lossy_construct and szwf.lossy_reconstruct)
        subfiles.path2file = subfiles.path2file.substr(0, subfiles.path2file.rfind('.'));
    auto basename = subfiles.path2file.substr(subfiles.path2file.rfind('/') + 1);

    if (opath.empty()) opath = input_path.empty() ? opath = "" : opath = input_path;
    opath += "/";

    // zip
    subfiles.compress.huff_base   = opath + basename;
    subfiles.compress.out_quant   = opath + basename + ".quant";
    subfiles.compress.out_outlier = opath + basename + ".outlier";
    subfiles.compress.out_yamp    = opath + basename + ".yamp";

    // unzip
    subfiles.decompress.in_yamp    = subfiles.path2file + ".yamp";
    subfiles.decompress.in_quant   = subfiles.path2file + ".quant";
    subfiles.decompress.in_outlier = subfiles.path2file + ".outlier";
    subfiles.decompress.out_xdata  = opath + basename + ".szx";
}
