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
#include <cstdlib>
#include <iostream>
#include <regex>
#include <string>
#include "argument_parser/document.hh"
#include "format.hh"

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
    if (cx_path2file.empty()) {
        cerr << log_err << "Not specifying input file!" << endl;
        to_abort = true;
    }
    if (d0 * d1 * d2 * d3 == 1 and not use_demo) {
        cerr << log_err << "Wrong input size(s)!" << endl;
        to_abort = true;
    }
    if (!to_encode and !to_decode and !to_dryrun) {
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

    if (to_dryrun and to_encode and to_decode) {
        cerr << log_warn << "No need to dry-run, encode and decode at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        to_encode = false;
        to_decode = false;
    }
    else if (to_dryrun and to_encode) {
        cerr << log_warn << "No need to dry-run and encode at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        to_encode = false;
    }
    else if (to_dryrun and to_decode) {
        cerr << log_warn << "No need to dry-run and decode at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        to_decode = false;
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
    if (cx_path2file.empty()) {
        cerr << log_err << "Not specifying input file!" << endl;
        to_abort = true;
    }
    if (d0 * d1 * d2 * d3 == 1 and not use_demo) {
        if (this->to_archive or this->to_dryrun) {
            cerr << log_err << "Wrong input size(s)!" << endl;
            to_abort = true;
        }
    }
    if (!to_archive and !to_extract and !to_dryrun) {
        cerr << log_err << "Select compress (-a), decompress (-x) or dry-run (-r)!" << endl;
        to_abort = true;
    }
    if (dtype != "f32" and dtype != "f64") {
        if (this->to_archive or this->to_dryrun) {
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

    if (to_dryrun and to_archive and to_extract) {
        cerr << log_warn << "No need to dry-run, compress and decompress at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        to_archive = false;
        to_extract = false;
    }
    else if (to_dryrun and to_archive) {
        cerr << log_warn << "No need to dry-run and compress at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        to_archive = false;
    }
    else if (to_dryrun and to_extract) {
        cerr << log_warn << "No need to dry-run and decompress at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        to_extract = false;
    }

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

ArgPack::ArgPack(int argc, char** argv, bool huffman)
{
    if (argc == 1) {
        HuffmanDoc();
        exit(0);
    }
    // default values
    dict_size        = 1024;
    input_rep        = 2;
    huffman_datalen  = -1;  // TODO argcheck
    huff_byte        = 4;
    huffman_chunk    = 512;
    read_args_status = 0;

    n_dim = -1;
    d0    = 1;
    d1    = 1;
    d2    = 1;
    d3    = 1;

    to_encode = false;
    to_decode = false;

    get_entropy = false;

    to_dryrun = false;  // TODO dry run is meaningful differently for cuSZ and Huffman

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
                    if (string(argv[i]) == "--help") goto _HELP;
                    // major features
                    if (string(argv[i]) == "--enc" or string(argv[i]) == "--encode") goto _ENCODE;
                    if (string(argv[i]) == "--dec" or string(argv[i]) == "--decode") goto _DECODE;
                    if (string(argv[i]) == "--dry-run") goto _DRY_RUN;
                    if (string(argv[i]) == "--verify") goto _VERIFY;
                    if (string(argv[i]) == "--input") goto _INPUT_DATUM;
                    if (string(argv[i]) == "--entropy") goto _ENTROPY;
                    if (string(argv[i]) == "--input-rep" or string(argv[i]) == "--interpret") goto _REP;
                    if (string(argv[i]) == "--huffman-rep") goto _HUFF_BYTE;
                    if (string(argv[i]) == "--huffman-chunk") goto _HUFF_CHUNK;
                    if (string(argv[i]) == "--dict-size") goto _DICT;
                    if (string(argv[i]) == "--gzip") {
                        to_gzip = true;
                        break;
                    }  // wenyu: if there is "--gzip", set member field to_gzip true
                // work
                // ----------------------------------------------------------------
                case 'e':
                _ENCODE:
                    to_encode = true;
                    break;
                case 'd':
                _DECODE:
                    to_decode = true;
                    break;
                case 'V':
                _VERIFY:
                    verify_huffman = true;  // TODO verify huffman in workflow
                    break;
                case 'r':
                _DRY_RUN:
                    // dry-run
                    to_dryrun = true;
                    break;
                case 'E':
                _ENTROPY:
                    get_entropy = true;
                    break;
                // input dimensionality
                // ----------------------------------------------------------------
                case '1':
                    n_dim = 1;
                    if (i + 1 <= argc) {
                        d0              = str2int(argv[++i]);
                        huffman_datalen = d0;
                    }
                    break;
                case '2':
                    n_dim = 2;
                    if (i + 2 <= argc) {
                        d0 = str2int(argv[++i]), d1 = str2int(argv[++i]);
                        huffman_datalen = d0 * d1;
                    }
                    break;
                case '3':
                    n_dim = 3;
                    if (i + 3 <= argc) {
                        d0 = str2int(argv[++i]), d1 = str2int(argv[++i]), d2 = str2int(argv[++i]);
                        huffman_datalen = d0 * d1 * d2;
                    }
                    break;
                case '4':
                    n_dim = 4;
                    if (i + 4 <= argc) {
                        d0 = str2int(argv[++i]), d1 = str2int(argv[++i]);
                        d2 = str2int(argv[++i]), d3 = str2int(argv[++i]);
                        huffman_datalen = d0 * d1 * d2 * d3;
                    }
                    break;
                // help document
                // ----------------------------------------------------------------
                case 'h':
                _HELP:
                    HuffmanDoc();
                    exit(0);
                    break;
                // input datum file
                // ----------------------------------------------------------------
                case 'i':
                _INPUT_DATUM:
                    if (i + 1 <= argc) { cx_path2file = string(argv[++i]); }
                    break;
                case 'R':
                _REP:
                    if (i + 1 <= argc) input_rep = str2int(argv[++i]);
                    break;
                case 'H':
                _HUFF_BYTE:
                    if (i + 1 <= argc) huff_byte = str2int(argv[++i]);
                    break;
                case 'C':
                _HUFF_CHUNK:
                    if (i + 1 <= argc) huffman_chunk = str2int(argv[++i]);
                    break;
                case 'c':
                _DICT:
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

ArgPack::ArgPack(int argc, char** argv)
{
    if (argc == 1) {
        cuszDoc();
        exit(0);
    }
    // default values
    dict_size              = 1024;
    quant_byte             = 2;
    huff_byte              = 4;
    huffman_chunk          = 512;
    n_dim                  = -1;
    d0                     = 1;
    d1                     = 1;
    d2                     = 1;
    d3                     = 1;
    mantissa               = 1.23;
    exponent               = -4.56;
    to_archive             = false;
    to_extract             = false;
    use_demo               = false;
    verbose                = false;
    to_verify              = false;
    verify_huffman         = false;
    skip_huffman           = false;
    skip_writex            = false;
    pre_binning            = false;
    to_dryrun              = false;
    autotune_huffman_chunk = true;
    read_args_status       = 0;

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
                    if (string(argv[i]) == "--help") goto _HELP;              // DOCUMENT
                    if (string(argv[i]) == "--version") goto _VERSION;        //
                    if (string(argv[i]) == "--verbose") goto _VERBOSE;        //
                    if (string(argv[i]) == "--entropy") goto _ENTROPY;        //
                    if (string(argv[i]) == "--meta") goto _META;              //
                    if (string(argv[i]) == "--mode") goto _MODE;              // COMPRESSION CONFIG
                    if (string(argv[i]) == "--quant-byte") goto _QUANT_BYTE;  //
                    if (string(argv[i]) == "--huff-byte") goto _HUFF_BYTE;    //
                    if (string(argv[i]) == "--huff-chunk") goto _HUFF_CHUNK;  //
                    if (string(argv[i]) == "--eb") goto _ERROR_BOUND;         //
                    if (string(argv[i]) == "--dict-size") goto _DICT;         //
                    if (string(argv[i]) == "--dtype") goto _TYPE;             //
                    if (string(argv[i]) == "--input") goto _INPUT_DATUM;      // INPUT
                    if (string(argv[i]) == "--demo") goto _DEMO;              //
                    if (string(argv[i]) == "--verify") goto _VERIFY;          //
                    if (string(argv[i]) == "--len") goto _LEN;                //
                    if (string(argv[i]) == "--compress") goto _COMPRESS;      // WORKFLOW
                    if (string(argv[i]) == "--zip") goto _COMPRESS;           //
                    if (string(argv[i]) == "--decompress") goto _DECOMPRESS;  //
                    if (string(argv[i]) == "--unzip") goto _DECOMPRESS;       //
                    if (string(argv[i]) == "--dry-run") goto _DRY_RUN;        //
                    if (string(argv[i]) == "--skip") goto _EXCLUDE;           //
                    if (string(argv[i]) == "--exclude") goto _EXCLUDE;        //
                    if (string(argv[i]) == "--pre") goto _PRE;                // IO
                    if (string(argv[i]) == "--output") goto _XOUT;            //
                    if (string(argv[i]) == "--opath") {  // TODO the followings has no single-letter options
                        if (i + 1 <= argc)
                            this->opath = string(argv[++i]);  // TODO does not apply for preprocessed such as binning
                        break;
                    }
                    if (string(argv[i]) == "--origin") {
                        if (i + 1 <= argc) this->x_fi_origin = string(argv[++i]);
                        break;
                    }
                    if (string(argv[i]) == "--gzip") {
                        to_gzip = true;
                        break;  // wenyu: if there is "--gzip", set member field to_gzip true
                    }
                    // if (string(argv[i]) == "--coname") {
                    //     // TODO does not apply for preprocessed such as binning
                    //     if (i + 1 <= argc) ap->coname = string(argv[++i]);
                    //     break;
                    // }
                    // if (string(argv[i]) == "--xoname") {
                    //     // TODO does not apply for preprocessed such as binning
                    //     if (i + 1 <= argc) ap->xoname = string(argv[++i]);
                    //     break;
                    // }
                // WORKFLOW
                case 'z':
                _COMPRESS:
                    to_archive = true;
                    break;
                case 'x':
                _DECOMPRESS:
                    to_extract = true;
                    break;
                case 'r':
                _DRY_RUN:
                    // dry-run
                    to_dryrun = true;
                    break;
                // COMPRESSION CONFIG
                case 'm':  // mode
                _MODE:
                    if (i + 1 <= argc) mode = string(argv[++i]);
                    break;
                // analysis
                case 'E':
                _ENTROPY:
                    get_entropy = true;
                    break;
                // OTHER WORKFLOW
                case 'S':
                _EXCLUDE:
                    if (i + 1 <= argc) {
                        string exclude(argv[++i]);
                        if (exclude.find("huffman") != std::string::npos) skip_huffman = true;
                        if (exclude.find("write.x") != std::string::npos) skip_writex = true;
                    }
                    break;
                // INPUT
                case 'l':
                _LEN:
                    if (i + 1 <= argc) {
                        std::stringstream   datalen(argv[++i]);
                        std::vector<string> dims;
                        while (datalen.good()) {
                            string substr;
                            getline(datalen, substr, ',');
                            dims.push_back(substr);
                        }
                        n_dim = dims.size();
                        if (n_dim == 1) {  //
                            d0 = str2int(dims[0].c_str());
                        }
                        if (n_dim == 2) {  //
                            d0 = str2int(dims[0].c_str()), d1 = str2int(dims[1].c_str());
                        }
                        if (n_dim == 3) {
                            d0 = str2int(dims[0].c_str()), d1 = str2int(dims[1].c_str());
                            d2 = str2int(dims[2].c_str());
                        }
                        if (n_dim == 4) {
                            d0 = str2int(dims[0].c_str()), d1 = str2int(dims[1].c_str());
                            d2 = str2int(dims[2].c_str()), d3 = str2int(dims[3].c_str());
                        }
                    }
                    break;
                case '1':
                    n_dim = 1;
                    if (i + 1 <= argc) {  //
                        d0 = str2int(argv[++i]);
                    }
                    break;
                case '2':
                    n_dim = 2;
                    if (i + 2 <= argc) {  //
                        d0 = str2int(argv[++i]), d1 = str2int(argv[++i]);
                    }
                    break;
                case '3':
                    n_dim = 3;
                    if (i + 3 <= argc) {
                        d0 = str2int(argv[++i]), d1 = str2int(argv[++i]);
                        d2 = str2int(argv[++i]);
                    }
                    break;
                case '4':
                    n_dim = 4;
                    if (i + 4 <= argc) {
                        d0 = str2int(argv[++i]), d1 = str2int(argv[++i]);
                        d2 = str2int(argv[++i]), d3 = str2int(argv[++i]);
                    }
                    break;
                case 'i':
                _INPUT_DATUM:
                    if (i + 1 <= argc) cx_path2file = string(argv[++i]);
                    break;
                    // alternative output
                case 'o':
                _XOUT:
                    cerr << log_err
                         << "\"-o\" will be working in the (near) future release. Pleae use \"--opath [path]\" "
                            "to "
                            "specify output path."
                         << endl;
                    exit(1);
                    break;
                // preprocess
                case 'p':
                _PRE:
                    if (i + 1 <= argc) {
                        string pre(argv[++i]);
                        if (pre.find("binning") != std::string::npos) pre_binning = true;
                    }
                    break;
                // demo datasets
                case 'D':
                _DEMO:
                    if (i + 1 <= argc) {
                        use_demo     = true;  // for skipping checking dimension args
                        demo_dataset = string(argv[++i]);
                    }
                    break;
                // DOCUMENT
                case 'h':
                _HELP:
                    cuszFullDoc();
                    exit(0);
                    break;
                case 'v':
                _VERSION:
                    // TODO
                    cout << log_info << version_text << endl;
                    break;
                // COMPRESSION CONFIG
                case 't':
                _TYPE:
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
                _META:  // TODO print .sz archive metadata
                    break;
                // internal representation and size
                case 'Q':
                _QUANT_BYTE:
                    if (i + 1 <= argc) quant_byte = str2int(argv[++i]);
                    break;
                case 'H':
                _HUFF_BYTE:
                    if (i + 1 <= argc) huff_byte = str2int(argv[++i]);
                    break;
                case 'C':
                _HUFF_CHUNK:
                    if (i + 1 <= argc) {  //
                        huffman_chunk          = str2int(argv[++i]);
                        autotune_huffman_chunk = false;
                    }
                    break;
                case 'e':
                _ERROR_BOUND:
                    if (i + 1 <= argc) {
                        string eb(argv[++i]);
                        if (eb.find('e') != std::string::npos) {
                            string dlm = "e";
                            // mantissa   = str2fp(eb.substr(0, eb.find(dlm)).c_str());
                            // exponent   = str2fp(eb.substr(eb.find(dlm) + dlm.length(), eb.length()).c_str());
                            mantissa = ::atof(eb.substr(0, eb.find(dlm)).c_str());
                            exponent = ::atof(eb.substr(eb.find(dlm) + dlm.length(), eb.length()).c_str());
                        }
                        else {
                            mantissa = ::atof(eb.c_str());
                            exponent = 0.0;
                        }
                    }
                    break;
                case 'y':
                _VERIFY:
                    if (i + 1 <= argc) {
                        string veri(argv[++i]);
                        if (veri.find("huffman") != std::string::npos) verify_huffman = true;
                        // TODO verify data quality
                    }
                    break;
                case 'V':
                _VERBOSE:
                    verbose = true;
                    break;
                case 'd':
                _DICT:
                    if (i + 1 <= argc) dict_size = str2int(argv[++i]);
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
    auto cx_input_path = cx_path2file.substr(0, cx_path2file.rfind("/") + 1);
    if (!to_archive && to_extract) cx_path2file = cx_path2file.substr(0, cx_path2file.rfind("."));
    auto cx_basename = cx_path2file.substr(cx_path2file.rfind("/") + 1);

    if (opath == "") opath = cx_input_path == "" ? opath = "" : opath = cx_input_path;
    opath += "/";

    // zip
    c_huff_base  = opath + cx_basename;
    c_fo_q       = opath + cx_basename + ".quant";
    c_fo_outlier = opath + cx_basename + ".outlier";
    c_fo_yamp    = opath + cx_basename + ".yamp";

    // unzip
    x_fi_yamp    = cx_path2file + ".yamp";
    x_fi_q       = cx_path2file + ".quant";
    x_fi_outlier = cx_path2file + ".outlier";
    x_fo_xd      = opath + cx_basename + ".szx";
}