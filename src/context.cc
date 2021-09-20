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

#include <cassert>
#include <cmath>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include "argument_parser/document.hh"
#include "context.hh"
#include "utils/format.hh"
#include "utils/strhelper.hh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

// TODO check version
const char* VERSION_TEXT  = "2021-09-19.1";
const int   VERSION       = 202107132;
const int   COMPATIBILITY = 0;

namespace {

void set_preprocess(cuszCTX* ctx, const char* in_str)
{
    str_list opts;
    StrHelper::parse_strlist(in_str, opts);

    for (auto k : opts) {
        // TODO
    }
}

void set_report(cuszCTX* ctx, const char* in_str)
{
    str_list opts;
    StrHelper::parse_strlist(in_str, opts);

    for (auto o : opts) {
        if (StrHelper::is_kv_pair(o)) {
            auto kv = StrHelper::parse_kv_onoff(o);

            if (kv.first == "quality")
                ctx->report.quality = kv.second;
            else if (kv.first == "cr")
                ctx->report.cr = kv.second;
            else if (kv.first == "compressibility")
                ctx->report.compressibility = kv.second;
            else if (kv.first == "time")
                ctx->report.time = kv.second;
        }
        else {
            if (o == "quality")
                ctx->report.quality = true;
            else if (o == "cr")
                ctx->report.cr = true;
            else if (o == "compressibility")
                ctx->report.compressibility = true;
            else if (o == "time")
                ctx->report.time = true;
        }
    }
}

void set_config(cuszCTX* ctx, const char* in_str)
{
    map_t opts;
    StrHelper::parse_strlist_as_kv(in_str, opts);

    for (auto kv : opts) {
        if (kv.first == "mode") { ctx->mode = std::string(kv.second); }
        else if (kv.first == "eb") {
            ctx->eb = StrHelper::str2fp(kv.second);
        }
        else if (kv.first == "cap") {
            ctx->dict_size = StrHelper::str2int(kv.second);
            ctx->radius    = ctx->dict_size / 2;
        }
        else if (kv.first == "huffbyte") {
            ctx->huff_nbyte = StrHelper::str2int(kv.second);
        }
        else if (kv.first == "quantbyte") {
            ctx->quant_nbyte = StrHelper::str2int(kv.second);
        }
        else if (kv.first == "huffchunk") {
            ctx->huffman_chunk              = StrHelper::str2int(kv.second);
            ctx->task_is.autotune_huffchunk = false;
        }
        else if (kv.first == "demo") {
            ctx->task_is.use_demo_dataset = true;
            ctx->demo_dataset             = string(kv.second);
            ctx->load_demo_sizes();
        }
        else if (kv.first == "predictor") {
            ctx->task_is.predictor = string(kv.second);
        }

        // when to enable anchor
        if (ctx->task_is.predictor == "spline3") ctx->task_is.use_anchor = true;
        if ((kv.first == "anchor") and  //
            (string(kv.second) == "on" or string(kv.second) == "ON"))
            ctx->task_is.use_anchor = true;
    }
}

}  // namespace

void cuszCTX::load_demo_sizes()
{
    const std::unordered_map<std::string, std::vector<int>> dataset_entries = {
        {std::string("hacc"), {280953867, 1, 1, 1, 1}},    {std::string("hacc1b"), {1073726487, 1, 1, 1, 1}},
        {std::string("cesm"), {3600, 1800, 1, 1, 2}},      {std::string("hurricane"), {500, 500, 100, 1, 3}},
        {std::string("nyx-s"), {512, 512, 512, 1, 3}},     {std::string("nyx-m"), {1024, 1024, 1024, 1, 3}},
        {std::string("qmc"), {288, 69, 7935, 1, 3}},       {std::string("qmcpre"), {69, 69, 33120, 1, 3}},
        {std::string("exafel"), {388, 59200, 1, 1, 2}},    {std::string("rtm"), {235, 849, 849, 1, 3}},
        {std::string("parihaka"), {1168, 1126, 922, 1, 3}}};

    if (not demo_dataset.empty()) {
        auto f = dataset_entries.find(demo_dataset);
        if (f == dataset_entries.end()) throw std::runtime_error("no such dataset as" + demo_dataset);
        auto demo_xyzw = f->second;

        x = demo_xyzw[0];
        y = demo_xyzw[1];
        z = demo_xyzw[2];
        w = demo_xyzw[3];

        ndim = demo_xyzw[4];
    }

    data_len = x * y * z * w;
}

void cuszCTX::trap(int _status) { this->read_args_status = _status; }

void cuszCTX::check_args()
{
    bool to_abort = false;
    if (fnames.path2file.empty()) {
        cerr << LOG_ERR << "must specify input file" << endl;
        to_abort = true;
    }

    if (data_len == 1 and not task_is.use_demo_dataset) {
        if (task_is.construct or task_is.dryrun) {
            cerr << LOG_ERR << "wrong input size" << endl;
            to_abort = true;
        }
    }
    if (not task_is.construct and not task_is.reconstruct and not task_is.dryrun) {
        cerr << LOG_ERR << "select compress (-z), decompress (-x) or dry-run (-r)" << endl;
        to_abort = true;
    }
    if (dtype != "f32" and dtype != "f64") {
        if (task_is.construct or task_is.dryrun) {
            cout << dtype << endl;
            cerr << LOG_ERR << "must specify data type" << endl;
            to_abort = true;
        }
    }

    if (quant_nbyte == 1) {  // TODO
        assert(dict_size <= 256);
    }
    else if (quant_nbyte == 2) {
        assert(dict_size <= 65536);
    }

    if (task_is.dryrun and task_is.construct and task_is.reconstruct) {
        cerr << LOG_WARN << "no need to dry-run, compress and decompress at the same time" << endl;
        cerr << LOG_WARN << "dryrun only" << endl << endl;
        task_is.construct   = false;
        task_is.reconstruct = false;
    }
    else if (task_is.dryrun and task_is.construct) {
        cerr << LOG_WARN << "no need to dry-run and compress at the same time" << endl;
        cerr << LOG_WARN << "dryrun only" << endl << endl;
        task_is.construct = false;
    }
    else if (task_is.dryrun and task_is.reconstruct) {
        cerr << LOG_WARN << "no need to dry-run and decompress at the same time" << endl;
        cerr << LOG_WARN << "will dryrun only" << endl << endl;
        task_is.reconstruct = false;
    }

    // if (task_is.gtest) {
    //     if (task_is.dryrun) { task_is.gtest = false; }
    //     else {
    //         if (not(task_is.construct and task_is.reconstruct)) { task_is.gtest = false; }
    //         if (fnames.origin_cmp == "") { task_is.gtest = false; }
    //     }
    // }

    if (to_abort) {
        print_short_doc();
        exit(-1);
    }
}

void cuszCTX::print_short_doc()
{
    cout << "\n>>>>  cusz build: " << VERSION_TEXT << "\n";
    cout << cusz_short_doc << endl;
}

void cuszCTX::print_full_doc()
{
    cout << "\n>>>>  cusz build: " << VERSION_TEXT << "\n";
    cout << StrHelper::doc_format(cusz_full_doc) << endl;
}

cuszCTX::cuszCTX(int argc, char** argv)
{
    if (argc == 1) {
        print_short_doc();
        exit(0);
    }

    opath = "";

    int i = 1;
    while (i < argc) {
        if (argv[i][0] == '-') {
            auto long_opt = string(argv[i]);
            switch (argv[i][1]) {
                // ----------------------------------------------------------------
                case '-':
                    // string list
                    if (long_opt == "--config") goto tag_config;
                    if (long_opt == "--report") goto tag_report;
                    if (long_opt == "--skip") {
                        if (i + 1 <= argc) {
                            string exclude(argv[++i]);
                            if (exclude.find("huffman") != std::string::npos) { task_is.skip_huffman = true; }
                            if (exclude.find("write2disk") != std::string::npos) { task_is.skip_write2disk = true; }
                        }
                        break;
                    }

                    if (long_opt == "--help") goto tag_help;              // DOCUMENT
                    if (long_opt == "--version") goto tag_version;        //
                    if (long_opt == "--predictor") goto tag_predictor;    //
                    if (long_opt == "--meta") goto tag_meta;              //
                    if (long_opt == "--mode") goto tag_mode;              // COMPRESSION CONFIG
                    if (long_opt == "--eb") goto tag_error_bound;         //
                    if (long_opt == "--dtype") goto tag_type;             //
                    if (long_opt == "--input") goto tag_input;            // INPUT
                    if (long_opt == "--len") goto tag_len;                //
                    if (long_opt == "--compress") goto tag_compress;      // WORKFLOW
                    if (long_opt == "--zip") goto tag_compress;           //
                    if (long_opt == "--decompress") goto tag_decompress;  //
                    if (long_opt == "--unzip") goto tag_decompress;       //
                    if (long_opt == "--dry-run") goto tag_dryrun;         //
                    if (long_opt == "--pre") goto tag_preproc;            // IO
                    if (long_opt == "--analysis") goto tag_analysis;      //
                    if (long_opt == "--output") goto tag_x_out;           //
                    if (long_opt == "--verbose") goto tag_verbose;        //

                    if (long_opt == "--demo") {
                        if (i + 1 <= argc) {
                            task_is.use_demo_dataset = true;
                            demo_dataset             = string(argv[++i]);
                            load_demo_sizes();
                        }
                        break;
                    }

                    if (long_opt == "--opath") {  // TODO the followings has no single-letter options
                        if (i + 1 <= argc)
                            this->opath = string(argv[++i]);  // TODO does not apply for preprocessed such as binning
                        break;
                    }
                    if (long_opt == "--origin" or long_opt == "--compare") {
                        if (i + 1 <= argc) fnames.origin_cmp = string(argv[++i]);
                        break;
                    }
                    if (long_opt == "--gzip") {
                        task_is.lossless_gzip = true;
                        break;  // wenyu: if there is "--gzip", set member field to_gzip true
                    }
                    if (long_opt == "--nvcomp") {
                        throw std::runtime_error(
                            "[argparse] nvcomp is disabled temporarily in favor of code refactoring.");
                        task_is.lossless_nvcomp_cascade = false;
                        break;
                    }
                    if (long_opt == "--gtest") {
                        throw std::runtime_error(
                            "[argparse] gtest is disabled temporarily in favor of code refactoring.");
                        task_is.gtest = false;
                        break;
                    }
                // WORKFLOW
                case 'z':
                tag_compress:
                    task_is.construct = true;
                    break;
                case 'x':
                tag_decompress:
                    task_is.reconstruct = true;
                    break;
                case 'r':
                tag_dryrun:
                    task_is.dryrun = true;
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
                        string analysis(argv[++i]);
                        if (analysis.find("export-codebook") != std::string::npos) { task_is.export_book = true; }
                        if (analysis.find("export-quant") != std::string::npos) { task_is.export_quant = true; }
                    }
                    break;
                // INPUT
                case 'l':
                tag_len:
                    if (i + 1 <= argc) {
                        std::vector<string> dims;
                        ConfigHelper::parse_length_literal(argv[++i], dims);
                        ndim = dims.size();
                        y = z = w = 1;
                        x         = StrHelper::str2int(dims[0].c_str());
                        if (ndim >= 2) y = StrHelper::str2int(dims[1].c_str());
                        if (ndim >= 3) z = StrHelper::str2int(dims[2].c_str());
                        if (ndim >= 4) w = StrHelper::str2int(dims[3].c_str());
                        data_len = x * y * z * w;
                    }
                    break;
                case 'i':
                tag_input:
                    if (i + 1 <= argc) fnames.path2file = string(argv[++i]);
                    break;
                case 'p':
                tag_predictor:
                    if (i + 1 <= argc) { task_is.predictor = string(argv[++i]); }
                // alternative output
                case 'o':
                tag_x_out:
                    cerr << LOG_ERR
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
                        if (pre.find("binning") != std::string::npos) { task_is.pre_binning = true; }
                    }
                    break;
                // interactive mode
                case 'I':
                tag_interactive:
                    break;
                case 'c':
                tag_config:
                    if (i + 1 <= argc) set_config(this, argv[++i]);
                    break;
                // report
                case 'R':
                tag_report:
                    if (i + 1 <= argc) set_report(this, argv[++i]);
                    break;
                case 'V':
                tag_verbose:
                    verbose = true;
                    break;
                // DOCUMENT
                case 'h':
                tag_help:
                    print_full_doc();
                    exit(0);
                case 'v':
                tag_version:
                    cout << ">>>>  cusz build: " << VERSION_TEXT << "\n";
                    exit(0);
                // COMPRESSION CONFIG
                case 't':
                tag_type:
                    if (i + 1 <= argc) {
                        string s = string(string(argv[++i]));
                        if (s == "f32" or s == "fp4")
                            dtype = "f32";
                        else if (s == "f64" or s == "fp8")
                            dtype = "f64";
                    }
                    break;
                case 'M':
                tag_meta:  // TODO print .cusza archive metadat
                    break;
                case 'e':
                tag_error_bound:
                    if (i + 1 <= argc) {
                        char* end;
                        this->eb = std::strtod(argv[++i], &end);
                    }
                    break;
                default:
                    const char* notif_prefix = "invalid option value at position ";
                    char*       notif;
                    int         size = asprintf(&notif, "%d: %s", i, argv[i]);
                    cerr << LOG_ERR << notif_prefix << "\e[1m" << notif << "\e[0m"
                         << "\n";
                    cerr << string(LOG_NULL.length() + strlen(notif_prefix), ' ');
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
            cerr << LOG_ERR << notif_prefix << "\e[1m" << notif
                 << "\e[0m"
                    "\n"
                 << string(LOG_NULL.length() + strlen(notif_prefix), ' ')  //
                 << "\e[1m"                                                //
                 << string(strlen(notif), '~')                             //
                 << "\e[0m\n";
            trap(-1);
        }
        i++;
    }

    // phase 1: check grammar
    if (read_args_status != 0) {
        cout << LOG_INFO << "Exiting..." << endl;
        // after printing ALL argument errors
        exit(-1);
    }

    // phase 2: check if meaningful
    check_args();
    // phase 3: sort out filenames
    sort_out_fnames();
}

void cuszCTX::sort_out_fnames()
{
    // (1) "fname"          -> "", "fname"
    // (2) "./fname"        -> "./" "fname"
    // (3) "/path/to/fname" -> "/path/to", "fname"
    auto input_path = fnames.path2file.substr(0, fnames.path2file.rfind('/') + 1);
    if (not task_is.construct and task_is.reconstruct)
        fnames.path2file = fnames.path2file.substr(0, fnames.path2file.rfind('.'));
    fnames.basename = fnames.path2file.substr(fnames.path2file.rfind('/') + 1);

    if (opath.empty()) opath = input_path.empty() ? opath = "" : opath = input_path;
    opath += "/";

    fnames.path_basename = opath + fnames.basename;
}
