//
// Created by jtian on 4/24/20.
//

#include "argparse.hh"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <regex>
#include <string>
#include "format.hh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

// TODO update with default values

// TODO check version
const char* version_text  = "version: pre-alpha, build: 2020-04-30";
const int   version       = 200430;
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
    if (fname.empty()) {
        cerr << log_err << "Not specifying input file!" << endl;
        to_abort = true;
    }
    if (d0 * d1 * d2 * d3 == 1 and not use_demo) {
        cerr << log_err << "Wrong input size(s)!" << endl;
        to_abort = true;
    }
    if (!to_encode and !to_decode and !dry_run) {
        cerr << log_err << "Select encode (-a), decode (-x) or dry-run (-r)!" << endl;
        to_abort = true;
    }
    // if (dtype != "f32" and dtype != "f64") {
    //     cout << dtype << endl;
    //     cerr << log_err << "Not specifying data type!" << endl;
    //     to_abort = true;
    // }

    if (input_rep == 8) {  // TODO
        assert(dict_size <= 256);
    }
    else if (input_rep == 16) {
        assert(dict_size <= 65536);
    }

    if (dry_run and to_encode and to_decode) {
        cerr << log_warn << "No need to dry-run, encode and decode at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        to_encode = false;
        to_decode = false;
    }
    else if (dry_run and to_encode) {
        cerr << log_warn << "No need to dry-run and encode at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        to_encode = false;
    }
    else if (dry_run and to_decode) {
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
    if (fname.empty()) {
        cerr << log_err << "Not specifying input file!" << endl;
        to_abort = true;
    }
    if (d0 * d1 * d2 * d3 == 1 and not use_demo) {
        cerr << log_err << "Wrong input size(s)!" << endl;
        to_abort = true;
    }
    if (!to_archive and !to_extract and !dry_run) {
        cerr << log_err << "Select compress (-a), decompress (-x) or dry-run (-r)!" << endl;
        to_abort = true;
    }
    if (dtype != "f32" and dtype != "f64") {
        cout << dtype << endl;
        cerr << log_err << "Not specifying data type!" << endl;
        to_abort = true;
    }

    if (quant_rep == 8) {  // TODO
        assert(dict_size <= 256);
    }
    else if (quant_rep == 16) {
        assert(dict_size <= 65536);
    }

    if (dry_run and to_archive and to_extract) {
        cerr << log_warn << "No need to dry-run, compress and decompress at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        to_archive = false;
        to_extract = false;
    }
    else if (dry_run and to_archive) {
        cerr << log_warn << "No need to dry-run and compress at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        to_archive = false;
    }
    else if (dry_run and to_extract) {
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
    string instruction =
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
    string instruction =
        "\n"
        "OVERVIEW: cuSZ: An Efficient GPU-Based Error-Bounded Lossy Compression Framework for Scientific Data\n"
        "\n"
        "USAGE:\n"
        "  The basic use with demo datum is listed below,\n"
        "    ./bin/cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -D cesm -z -x\n"
        "                 ^  ~~~~~~ ~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ~~~~~~~  ^  ^\n"
        "                 |   mode   error         input datum file        demo   |  |\n"
        "               dtype        bound                                 data  zip unzip\n"
        "\n"
        "  compress and extract, demo dataset:\n"
        "    cusz -f32|-f64 -m [eb mode] -e [eb] -i [datum file] -D [demo dataset] -z -x\n"
        "    (change \"-z -x\" to \"-r\" for dry run)\n"
        "  compress and extract, arbitrary datum:\n"
        "    cusz -f32|-f64 -m [eb mode] -e [eb] -i [datum file] -1|-2|-3 [nx [ny [nz]] -z -x\n"
        "    (change \"-z -x\" to \"-r\" for dry run)\n"
        "  \n"
        "EXAMPLES\n"
        "  CESM example:\n"
        "    ./bin/cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -D cesm -z -x\n"
        "    ./bin/cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -D cesm -r\n"
        "  Hurricane Isabel example:\n"
        "    ./bin/cusz -f32 -m r2r -e 1e-4 -i ./data/sample-hurr-CLOUDf48 -D hurricane -z -x\n"
        "    ./bin/cusz -f32 -m r2r -e 1e-4 -i ./data/sample-hurr-CLOUDf48 -D hurricane -r\n"
        "  EXAFEL example:\n"
        "    ./bin/cusz -f32 -m r2r -e 1e-4 -i ./data/sample-exafel-59200x388 -D exafeldemo -z -x --pre binning\n"
        "    ./bin/cusz -f32 -m r2r -e 1e-4 -i ./data/sample-exafel-59200x388 -D exafeldemo -z -x --pre binning --skip huffman\n"
        "\n"
        "DOC:\n"
        "  Type \"cusz -h\" for details.\n";
    cout << instruction << endl;
}

void  //
ArgPack::cuszFullDoc()
{
    string doc =
        "*NAME*\n"
        "        cuSZ: An Efficient GPU-Based Error-Bounded Lossy Compression Framework for Scientific Data\n"
        "        Lowercased \"*cusz*\" is the command."
        //"        cusz - a GPU-accelerated error-bounded lossy compressor for scientific data.\n"
        "\n"
        "*SYNOPSIS*\n"
        "        The basic use is listed below,\n"
        "        *cusz* *-f*32 *-m* r2r *-e* 1e-4 *-i* ./data/sample-cesm-CLDHGH *-2* 3600 1800 *-z -x*\n"
        // "        cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -2 3600 1800 -z -x\n"
        "               ^  ~~~~~~ ~~~~~~~ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ~~~~~~~~~~~~  ^  ^\n"
        "               |   mode   error        input datum file        low-to-high  |  |\n"
        "             dtype        bound                                order       zip unzip\n"
        "\n"
        "        *cusz* *-f*32|*-f*64 *-m* [eb mode] *-e* [eb] *-i* [datum file] *-D* [demo dataset] *-z* *-x*\n"
        "        *cusz* *-f*32|*-f*64 *-m* [eb mode] *-e* [eb] *-i* [datum file] *-1*|*-2*|*-3* [nx [ny [nz]] *-z* *-x*\n"
        "\n"
        "*OPTIONS*\n"
        "    *Mandatory*\n"
        "        *-z* or *--compress* or *--*@z@*ip*\n"
        "        *-x* or *--e*@x@*tract* or *--decompress* or *--unzip*\n"
        "        *-r* or *--dry-*@r@*un*\n"
        "                No lossless Huffman codec. Only to get data quality summary.\n"
        "                In addition, quant. rep. and dict. size are retained\n"
        "\n"
        "        *-m* or *--*@m@*ode* <abs|r2r>\n"
        "                Specify error-controling mode. Supported modes include:\n"
        "                _abs_: absolute mode, eb = input eb\n"
        "                _r2r_: relative-to-value-range mode, eb = input eb x value range\n"
        "\n"
        "        *-e* or *--eb* or *--error-bound* [num]\n"
        "                Specify error bound. e.g., _1.23_, _1e-4_, _1.23e-4.56_\n"
        "\n"
        "        *-i* or *--input* [datum file]\n"
        "        *-o* or *--output* [alternative decompressed file]\n"
        "                Specify otherwise decompressed file name.\n"
        "\n"
        "        *-d* or *--dict-size* [256|512|1024|...]\n"
        "                Specify dictionary size/quantization bin number.\n"
        "                Should be a power-of-2.\n"
        "\n"
        "        *-1* [x]       Specify 1D datum/field size.\n"
        "        *-2* [x] [y]   Specify 2D datum/field sizes, with dimensions from low to high.\n"
        "        *-3* [x] [y] [z]   Specify 3D datum/field sizes, with dimensions from low to high.\n"
        "\n"
        "    *Modules*\n"
        "        *-X* or *-S* or *--e*@x@*clude* or *--*@s@*kip* _module-1_,_module-2_,...,_module-n_,\n"
        "                Disable functionality modules. Supported module(s) include:\n"
        "                _huffman_  Huffman codec after prediction+quantization (p+q) and before reveresed p+q.\n"
        "                _write.x_  Skip write decompression data.\n"
        "\n"
        "        *-p* or *--pre* _method-1_,_method-2_,...,_method-n_\n"
        "                Enable preprocessing. Supported preproessing method(s) include:\n"
        "                _binning_  Downsampling datum by 2x2 to 1.\n"
        "\n"
        "    *Demonstration*\n"
        "        *-h* or *--help*   Get help documentation.\n"
        "\n"
        "        *-M* or *--meta*   Get archive metadata. (TODO)\n"
        "\n"
        "        *-D* or *--demo* [demo-dataset]\n"
        "                Use demo dataset, will omit given dimension(s). Supported datasets include:\n"
        "                1D: _hacc_    2D: _cesm_  _exafeldemo_\n"
        "                3D: _hurricane_  _nyx_  _qmc_  _qmcpre_  _aramco_\n"
        "\n"
        "    *Internal* (will be automated with configuration when going public)\n"
        "        *-Q* or *--*@q@*uant-rep* or *--bcode-bitwidth* <8|16|32>\n"
        "                Specify bincode/quantization code representation.\n"
        "                Options _8_, _16_, _32_ are for *uint8_t*, *uint16_t*, *uint32_t*, respectively.\n"
        "                ^^Manually specifying this may not result in optimal memory footprint.^^\n"
        "\n"
        "        *-H* or *--*@h@*uffman-rep* or *--hcode-bitwidth* <32|64>\n"
        "                Specify Huffman codeword representation.\n"
        "                Options _32_, _64_ are for *uint32_t*, *uint64_t*, respectively.\n"
        "                ^^Manually specifying this may not result in optimal memory footprint.^^\n"
        "\n"
        "        *-C* or *--huffman-*@c@*hunk* or *--hcode-chunk* [256|512|1024|...]\n"
        "                Specify chunk size for Huffman codec.\n"
        "                Should be a power-of-2 that is sufficiently large.\n"
        "                ^^This affects Huffman decoding performance significantly.^^\n"
        "\n"
        "*EXAMPLES*\n"
        "    *Demo Datasets*\n"
        "        *CESM* example:\n"
        "        ./bin/cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -D cesm -z -x\n"
        "        ./bin/cusz -f32 -m r2r -e 1e-4 -i ./data/sample-cesm-CLDHGH -D cesm -r\n"
        "\n"
        "        *Hurricane Isabel* example:\n"
        "        ./bin/cusz -f32 -m r2r -e 1e-4 -i ./data/sample-hurr-CLOUDf48 -D hurricane -z -x\n"
        "        ./bin/cusz -f32 -m r2r -e 1e-4 -i ./data/sample-hurr-CLOUDf48 -D hurricane -r\n"
        "\n"
        "        *EXAFEL* example:\n"
        "        ./bin/cusz -f32 -m r2r -e 1e-4 -i ./data/sample-exafel-59200x388 -D exafeldemo -z -x --pre binning\n"
        "        ./bin/cusz -f32 -m r2r -e 1e-4 -i ./data/sample-exafel-59200x388 -D exafeldemo -z -x --pre binning --skip huffman\n";

    cout << format(doc) << endl;
}

ArgPack::ArgPack(int argc, char** argv, bool huffman)
{
    if (argc == 1) {
        HuffmanDoc();
        exit(0);
    }
    // default values
    dict_size       = 1024;
    input_rep       = 16;
    huffman_datalen = -1;  // TODO argcheck
    huffman_rep     = 32;
    huffman_chunk   = 512;

    n_dim = -1;
    d0    = 1;
    d1    = 1;
    d2    = 1;
    d3    = 1;

    to_encode = false;
    to_decode = false;

    get_entropy = false;

    dry_run = false;  // TODO dry run is meaningful differently for cuSZ and Huffman

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
                    if (string(argv[i]) == "--huffman-rep") goto _HUFFMANCODE;
                    if (string(argv[i]) == "--huffman-chunk") goto _HUFFMANCHUNKSIZE;
                    if (string(argv[i]) == "--dict-size") goto _DICT;
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
                    dry_run = true;
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
                        d0 = str2int(argv[++i]), d1 = str2int(argv[++i]), d2 = str2int(argv[++i]), d3 = str2int(argv[++i]);
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
                    if (i + 1 <= argc) {
                        fname = string(argv[++i]);
                    }
                    break;
                case 'R':
                _REP:
                    if (i + 1 <= argc) input_rep = str2int(argv[++i]);
                    break;
                case 'H':
                _HUFFMANCODE:
                    if (i + 1 <= argc) huffman_rep = str2int(argv[++i]);
                    break;
                case 'C':
                _HUFFMANCHUNKSIZE:
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
    dict_size      = 1024;
    quant_rep      = 16;
    huffman_rep    = 32;
    huffman_chunk  = 512;
    n_dim          = -1;
    d0             = 1;
    d1             = 1;
    d2             = 1;
    d3             = 1;
    mantissa       = 1.23;
    exponent       = -4.56;
    to_archive     = false;
    to_extract     = false;
    use_demo       = false;
    verbose        = false;
    to_verify      = false;
    verify_huffman = false;
    skip_huffman   = false;
    skip_writex    = false;
    pre_binning    = false;
    dry_run        = false;

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
                    if (string(argv[i]) == "--entropy") goto _ENTROPY;
                    if (string(argv[i]) == "--version") goto _VERSION;
                    if (string(argv[i]) == "--verbose") goto _VERBOSE;
                    if (string(argv[i]) == "--mode") goto _MODE;
                    if (string(argv[i]) == "--input") goto _INPUT_DATUM;
                    if (string(argv[i]) == "--demo") goto _DEMO;
                    if (string(argv[i]) == "--quant-rep" or string(argv[i]) == "--bcode-bitwidth") goto _BINCODE;
                    if (string(argv[i]) == "--huffman-rep" or string(argv[i]) == "--hcode-bitwidth") goto _HUFFMANCODE;
                    if (string(argv[i]) == "--huffman-chunk" or string(argv[i]) == "--hcode-chunk") goto _HUFFMANCHUNKSIZE;
                    if (string(argv[i]) == "--eb" or string(argv[i]) == "--error-bound") goto _ERROR_BOUND;
                    if (string(argv[i]) == "--error-bound") goto _ERROR_BOUND;
                    if (string(argv[i]) == "--verify") goto _VERIFY;
                    if (string(argv[i]) == "--dict-size") goto _DICT;
                    if (string(argv[i]) == "--compress" or string(argv[i]) == "--zip") goto _COMPRESS;
                    if (string(argv[i]) == "--decompress" or string(argv[i]) == "--extract" or string(argv[i]) == "--unzip") goto _DECOMPRESS;
                    if (string(argv[i]) == "--exclude" or string(argv[i]) == "--skip") goto _EXCLUDE;
                    if (string(argv[i]) == "--dry-run") goto _DRY_RUN;
                    if (string(argv[i]) == "--meta") goto _META;
                    if (string(argv[i]) == "--pre") goto _PRE;
                    if (string(argv[i]) == "--output") goto _OUT;
                // work
                // ----------------------------------------------------------------
                case 'a':
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
                    dry_run = true;
                    break;
                case 'm':  // mode
                _MODE:
                    if (i + 1 <= argc) mode = string(argv[++i]);
                    break;
                    // analysis
                case 'E':
                _ENTROPY:
                    get_entropy = true;
                    break;
                // input dimensionality
                // ----------------------------------------------------------------
                case 'X':
                case 'S':
                _EXCLUDE:
                    if (i + 1 <= argc) {
                        string exclude(argv[++i]);
                        if (exclude.find("huffman") != std::string::npos) skip_huffman = true;
                        if (exclude.find("write.x") != std::string::npos) skip_writex = true;
                    }
                    break;
                // input dimensionality
                // ----------------------------------------------------------------
                case '1':
                    n_dim = 1;
                    if (i + 1 <= argc) {
                        d0 = str2int(argv[++i]);
                    }
                    break;
                case '2':
                    n_dim = 2;
                    if (i + 2 <= argc) {
                        d0 = str2int(argv[++i]);
                        d1 = str2int(argv[++i]);
                    }
                    break;
                case '3':
                    n_dim = 3;
                    if (i + 3 <= argc) {
                        d0 = str2int(argv[++i]);
                        d1 = str2int(argv[++i]);
                        d2 = str2int(argv[++i]);
                    }
                    break;
                case '4':
                    n_dim = 4;
                    if (i + 4 <= argc) {
                        d0 = str2int(argv[++i]);
                        d1 = str2int(argv[++i]);
                        d2 = str2int(argv[++i]);
                        d3 = str2int(argv[++i]);
                    }
                    break;
                // help document
                // ----------------------------------------------------------------
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
                // data type
                // ----------------------------------------------------------------
                case 'f':
                    if (string(argv[i]) == "-f32") dtype = "f32";
                    if (string(argv[i]) == "-f64") dtype = "f64";
                    break;
                // input datum file
                // ----------------------------------------------------------------
                case 'i':
                    //                    if (string(argv[i]) == "-i8") {
                    //                        dtype = "i8";
                    //                        break;
                    //                    }
                    //                    if (string(argv[i]) == "-i16") {
                    //                        dtype = "i16";
                    //                        break;
                    //                    }
                    //                    if (string(argv[i]) == "-i32") {
                    //                        dtype = "i32";
                    //                        break;
                    //                    }
                    //                    if (string(argv[i]) == "-i64") {
                    //                        dtype = "i64";
                    //                        break;
                    //                    }
                _INPUT_DATUM:
                    if (i + 1 <= argc) {
                        fname = string(argv[++i]);
                    }
                    break;
                    // alternative output
                case 'o':
                _OUT:
                    if (i + 1 <= argc) {
                        alt_xout_name = string(argv[++i]);
                    }
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
                // ----------------------------------------------------------------
                case 'D':
                _DEMO:
                    if (i + 1 <= argc) {
                        use_demo     = true;  // for skipping checking dimension args
                        demo_dataset = string(argv[++i]);
                    }
                    break;
                case 'M':
                _META:  // TODO print .sz archive metadata
                    break;
                // internal representation and size
                // ----------------------------------------------------------------
                case 'Q':
                _BINCODE:
                    if (i + 1 <= argc) quant_rep = str2int(argv[++i]);
                    break;
                case 'H':
                _HUFFMANCODE:
                    if (i + 1 <= argc) huffman_rep = str2int(argv[++i]);
                    break;
                case 'C':
                _HUFFMANCHUNKSIZE:
                    if (i + 1 <= argc) huffman_chunk = str2int(argv[++i]);
                    break;
                // error bound
                // ----------------------------------------------------------------
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
    CheckArgs();
}
