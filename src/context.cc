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
const char* VERSION_TEXT  = "2022-03-25.rc3";
const int   VERSION       = 20220325;
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

            if (kv.first == "cr")
                ctx->report.cr = kv.second;
            else if (kv.first == "compressibility")
                ctx->report.compressibility = kv.second;
            else if (kv.first == "time")
                ctx->report.time = kv.second;
        }
        else {
            if (o == "cr")
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

    auto is_enabled = [&](auto& v) -> bool { return v == "on" or v == "ON"; };

    for (auto kv : opts) {
        if (kv.first == "mode") { ctx->mode = std::string(kv.second); }
        else if (kv.first == "eb") {
            ctx->eb = StrHelper::str2fp(kv.second);
        }
        else if (kv.first == "cap") {  // to delete, only radius matters for compressor
            ctx->dict_size = StrHelper::str2int(kv.second);
            ctx->radius    = ctx->dict_size / 2;
        }
        else if (kv.first == "radius") {  // to adjust, only radiusn matters for compressor
            ctx->radius    = StrHelper::str2int(kv.second);
            ctx->dict_size = ctx->radius * 2;
        }
        else if (kv.first == "huffbyte") {
            ctx->huff_bytewidth = StrHelper::str2int(kv.second);
            ctx->codecs_in_use  = ctx->codec_force_fallback() ? 0b11 /*use both*/ : 0b01 /*use 4-byte*/;
        }
        else if (kv.first == "quantbyte") {
            ctx->quant_bytewidth = StrHelper::str2int(kv.second);
        }
        else if (kv.first == "huffchunk") {
            ctx->vle_sublen              = StrHelper::str2int(kv.second);
            ctx->use.autotune_vle_pardeg = false;
        }
        else if (kv.first == "demo") {
            ctx->use.predefined_demo = true;
            ctx->demo_dataset        = string(kv.second);
            ctx->load_demo_sizes();
        }
        else if (kv.first == "predictor") {
            ctx->predictor = string(kv.second);
        }
        else if (kv.first == "postcompress") {
            // TODO nvcomp, gzip, etc.
        }
        else if (kv.first == "anchor" and is_enabled(kv.second)) {
            ctx->use.anchor = true;
        }
        else if (kv.first == "releaseinput" and is_enabled(kv.second)) {
            ctx->use.release_input = true;
        }
        else if (kv.first == "pipeline") {
            ctx->compression_pipeline = kv.second;
        }
        else if (kv.first == "density") {  // refer to `SparseMethodSetup` in `config.hh`
            ctx->nz_density        = StrHelper::str2fp(kv.second);
            ctx->nz_density_factor = 1 / ctx->nz_density;
        }
        else if (kv.first == "densityfactor") {  // refer to `SparseMethodSetup` in `config.hh`
            ctx->nz_density_factor = StrHelper::str2fp(kv.second);
            ctx->nz_density        = 1 / ctx->nz_density_factor;
        }
        else if (kv.first == "gpuverify" and is_enabled(kv.second)) {
            ctx->use.gpu_verify = true;
        }

        // when to enable anchor
        if (ctx->predictor == "spline3") {
            // unconditionally use anchor when it is spline3
            ctx->use.anchor = true;
        }
    }
}

}  // namespace

/**
 * @deprecated
 *
 */
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

        x = demo_xyzw[0], y = demo_xyzw[1], z = demo_xyzw[2], w = demo_xyzw[3];
        ndim = demo_xyzw[4];
    }
    data_len = x * y * z * w;
}

void cuszCTX::trap(int _status) { this->read_args_status = _status; }

void cuszCTX::check_args_when_cli()
{
    bool to_abort = false;
    if (fname.fname.empty()) {
        cerr << LOG_ERR << "must specify input file" << endl;
        to_abort = true;
    }

    if (data_len == 1 and not use.predefined_demo) {
        if (task_is.construct or task_is.dryrun) {
            cerr << LOG_ERR << "wrong input size" << endl;
            to_abort = true;
        }
    }
    if (not task_is.construct and not task_is.reconstruct and not task_is.dryrun) {
        cerr << LOG_ERR << "select compress (-z), decompress (-x) or dry-run (-r)" << endl;
        to_abort = true;
    }
    if (false == ConfigHelper::check_dtype(dtype, false)) {
        if (task_is.construct or task_is.dryrun) {
            cout << dtype << endl;
            cerr << LOG_ERR << "must specify data type" << endl;
            to_abort = true;
        }
    }

    if (quant_bytewidth == 1)
        assert(dict_size <= 256);
    else if (quant_bytewidth == 2)
        assert(dict_size <= 65536);

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
                    if (long_opt == "--help") goto tag_help;              // DOCUMENT
                    if (long_opt == "--version") goto tag_version;        //
                    if (long_opt == "--predictor") goto tag_predictor;    //
                    if (long_opt == "--meta") goto tag_meta;              //
                    if (long_opt == "--mode") goto tag_mode;              // COMPRESSION CONFIG
                    if (long_opt == "--eb") goto tag_error_bound;         //
                    if (long_opt == "--predictor") goto tag_predictor;    //
                    if (long_opt == "--dtype") goto tag_type;             //
                    if (long_opt == "--input") goto tag_input;            // INPUT
                    if (long_opt == "--len") goto tag_len;                //
                    if (long_opt == "--compress") goto tag_compress;      // WORKFLOW
                    if (long_opt == "--zip") goto tag_compress;           //
                    if (long_opt == "--decompress") goto tag_decompress;  //
                    if (long_opt == "--unzip") goto tag_decompress;       //
                    if (long_opt == "--dry-run") goto tag_dryrun;         //
                    if (long_opt == "--pre") goto tag_preproc;            // IO
                    if (long_opt == "--output") goto tag_x_out;           //
                    if (long_opt == "--verbose") goto tag_verbose;        //

                    if (long_opt == "--pipeline") {
                        if (i + 1 <= argc) compression_pipeline == string(argv[++i]);
                        break;
                    }

                    if (long_opt == "--demo") {
                        if (i + 1 <= argc) {
                            use.predefined_demo = true;
                            demo_dataset        = string(argv[++i]);
                            load_demo_sizes();
                        }
                        break;
                    }

                    if (long_opt == "--skip") {
                        if (i + 1 <= argc) {
                            string exclude(argv[++i]);
                            if (exclude.find("huffman") != std::string::npos) { skip.huffman = true; }
                            if (exclude.find("write2disk") != std::string::npos) { skip.write2disk = true; }
                        }
                        break;
                    }
                    if (long_opt == "--export") {
                        // TODO
                        string extra_export(argv[++i]);
                        if (extra_export.find("codebook") != std::string::npos) { export_raw.book = true; }
                        if (extra_export.find("quant") != std::string::npos) { export_raw.quant = true; }
                    }

                    if (long_opt == "--opath") {  // TODO the followings has no single-letter options
                        if (i + 1 <= argc)
                            this->opath = string(argv[++i]);  // TODO does not apply for preprocessed such as binning
                        break;
                    }
                    if (long_opt == "--origin" or long_opt == "--compare") {
                        if (i + 1 <= argc) fname.origin_cmp = string(argv[++i]);
                        break;
                    }
                    if (long_opt == "--gzip") {
                        postcompress.cpu_gzip = true;
                        break;  // wenyu: if there is "--gzip", set member field to_gzip true
                    }
                    if (long_opt == "--nvcomp") {
                        throw std::runtime_error(
                            "[argparse] nvcomp is disabled temporarily in favor of code refactoring.");
                        postcompress.gpu_nvcomp_cascade = false;
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
                case 'p':
                tag_predictor:
                    if (i + 1 <= argc) predictor = string(argv[++i]);
                    break;
                case 'm':  // mode
                tag_mode:
                    if (i + 1 <= argc) {
                        mode = string(argv[++i]);
                        if (mode == "r2r") preprocess.prescan = true;
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
                    if (i + 1 <= argc) fname.fname = string(argv[++i]);
                    break;
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
                        if (pre.find("binning") != std::string::npos) { preprocess.binning = true; }
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

    // phase 1: check syntax
    if (read_args_status != 0) {
        cout << LOG_INFO << "Exiting..." << endl;
        // after printing ALL argument errors
        exit(-1);
    }

    // phase 2: check if legal
    check_args_when_cli();
    // phase 3: sort out filenames
    sort_out_fnames();
}

cuszCTX::cuszCTX(const char* config_str, bool dbg_print)
{
    /**
     **  >>> syntax
     **  comma-separated key-pairs
     **  "key1=val1,key2=val2[,...]"
     **
     **  >>> example
     **  "predictor=lorenzo,size=3600x1800"
     **
     **/

    // skip the default values
    // const char* ex_config = "size=3600x1800";

    using config_map_t = map_t;

    config_map_t opts;
    StrHelper::parse_strlist_as_kv(config_str, opts);

    for (auto kv : opts) {
        auto  k = kv.first;
        auto  v = kv.second;
        char* end;

        // general-mandatory (compress/decompress)
        if (k == "input") { fname.fname = string(v); }
        if (k == "do") {
            if (v == "dryrun") task_is.dryrun = true;
            if (v == "compress" or v == "zip") task_is.construct = true;
            if (v == "decompress" or v == "unzip") task_is.reconstruct = true;
        }

        // mandatory
        // compress
        if (k == "dtype" and ConfigHelper::check_dtype(v, false)) this->dtype = v;
        if (k == "errorbound" or k == "eb") eb = std::strtod(v.c_str(), &end);
        if (k == "mode" and ConfigHelper::check_cuszmode(v, true)) this->mode = v;
        if (k == "size") {
            std::vector<string> dims;
            ConfigHelper::parse_length_literal(v.c_str(), dims);
            ndim = dims.size();
            y = z = w = 1;
            x         = StrHelper::str2int(dims[0].c_str());
            if (ndim >= 2) y = StrHelper::str2int(dims[1].c_str());
            if (ndim >= 3) z = StrHelper::str2int(dims[2].c_str());
            if (ndim >= 4) w = StrHelper::str2int(dims[3].c_str());
            data_len = x * y * z * w;
        }
        if (k == "radius") { radius = StrHelper::str2int(v), dict_size = radius * 2; }
        if (k == "dictsize") { dict_size = StrHelper::str2int(v), radius = dict_size / 2; }
        if (k == "huffbyte") {
            huff_bytewidth = StrHelper::str2int(kv.second);
            codecs_in_use  = codec_force_fallback() ? 0b11 /*use both*/ : 0b01 /*use 4-byte*/;
        }

        // optional
        // decompress
        if (k == "origin" or k == "compare") { fname.origin_cmp = string(v); }

        // future use
        /*
        if (k == "predictor" and ConfigHelper::check_predictor(v, true)) {
            this->predictor = v;
            this->predictor     = ConfigHelper::predictor_lookup(v);
        }
        if (k == "codec" and ConfigHelper::check_codec(v, true)) {
            this->codec = v;  // TODO
            this->codec     = ConfigHelper::codec_lookup(v);
        }
        if (k == "spcodec" and ConfigHelper::check_codec(v, true)) {
            this->spcodec = v;  // TODO
            this->spcodec     = ConfigHelper::spcodec_lookup(v);
        }
        */
    }

    sort_out_fnames();

    if (dbg_print) {
        printf("\ninput config string:\n");
        printf("\n%s\n", config_str);

        for (auto kv : opts) {
            auto k = kv.first;
            auto v = kv.second;
            cout << k << "\t" << v << "\n";
        }
    }
}

void cuszCTX::sort_out_fnames()
{
    // (1) "fname"          -> "", "fname"
    // (2) "./fname"        -> "./" "fname"
    // (3) "/path/to/fname" -> "/path/to", "fname"
    auto input_path = fname.fname.substr(0, fname.fname.rfind('/') + 1);
    if (not task_is.construct and task_is.reconstruct) fname.fname = fname.fname.substr(0, fname.fname.rfind('.'));
    fname.basename = fname.fname.substr(fname.fname.rfind('/') + 1);

    if (opath.empty()) opath = input_path.empty() ? opath = "" : opath = input_path;
    opath += "/";

    fname.path_basename   = opath + fname.basename;
    fname.compress_output = fname.path_basename + ".cusza";
}
