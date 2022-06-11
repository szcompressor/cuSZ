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

#include "cli/document.hh"
#include "context.hh"

namespace cusz {
const char* VERSION_TEXT  = "2022-06-10.canary";
const int   VERSION       = 20220610;
const int   COMPATIBILITY = 0;
}  // namespace cusz

namespace {

void set_preprocess(cusz::context_t ctx, const char* in_str)
{
    str_list opts;
    StrHelper::parse_strlist(in_str, opts);

    for (auto k : opts) {
        // TODO
    }
}

void set_report(cusz::context_t ctx, const char* in_str)
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

void set_config(cusz::context_t ctx, const char* in_str, bool dbg_print = false)
{
    map_t opts;
    StrHelper::parse_strlist_as_kv(in_str, opts);

    if (dbg_print) {
        for (auto kv : opts) printf("%-*s %-s\n", 10, kv.first.c_str(), kv.second.c_str());
        std::cout << "\n";
    }

    std::string k, v;
    char*       end;

    auto optmatch   = [&](std::vector<std::string> vs) -> bool { return ConfigHelper::check_opt_in_list(k, vs); };
    auto is_enabled = [&](auto& v) -> bool { return v == "on" or v == "ON"; };

    for (auto kv : opts) {
        k = kv.first;
        v = kv.second;

        if (optmatch({"type", "dtype"})) {
            ConfigHelper::check_dtype(v, false);
            ctx->dtype = v;
        }
        else if (optmatch({"eb", "errorbound"})) {
            ctx->eb = StrHelper::str2fp(v);
        }
        else if (optmatch({"mode"})) {
            ConfigHelper::check_cuszmode(v, true);
            ctx->mode = v;
        }
        else if (optmatch({"len", "length"})) {
            cuszCTX::parse_input_length(v.c_str(), ctx);
        }
        else if (optmatch({"alloclen"})) {
            ctx->alloclen.len = StrHelper::str2int(v);
        }
        else if (optmatch({"demo"})) {
            ctx->use.predefined_demo = true;
            ctx->demo_dataset        = std::string(v);
            ctx->load_demo_sizes();
        }
        else if (optmatch({"cap", "booklen", "dictsize"})) {
            ctx->dict_size = StrHelper::str2int(v);
            ctx->radius    = ctx->dict_size / 2;
        }
        else if (optmatch({"radius"})) {
            ctx->radius    = StrHelper::str2int(v);
            ctx->dict_size = ctx->radius * 2;
        }
        else if (optmatch({"huffbyte"})) {
            ctx->huff_bytewidth = StrHelper::str2int(v);
            ctx->codecs_in_use  = ctx->codec_force_fallback() ? 0b11 /*use both*/ : 0b01 /*use 4-byte*/;
        }
        else if (optmatch({"huffchunk"})) {
            ctx->vle_sublen              = StrHelper::str2int(v);
            ctx->use.autotune_vle_pardeg = false;
        }
        else if (optmatch({"predictor"})) {
            ctx->predictor = std::string(v);
        }
        else if (optmatch({"codec"})) {
            // placeholder
        }
        else if (optmatch({"spcodec"})) {
            // placeholder
        }
        else if (optmatch({"anchor"}) and is_enabled(v)) {
            ctx->use.anchor = true;
        }
        else if (optmatch({"nondestructive"}) and is_enabled(v)) {
            // placeholder
        }
        else if (optmatch({"failfast"}) and is_enabled(v)) {
            // placeholder
        }
        else if (optmatch({"releaseinput"}) and is_enabled(v)) {
            ctx->use.release_input = true;
        }
        else if (optmatch({"pipeline"})) {
            ctx->pipeline = v;
        }
        else if (optmatch({"density"})) {  // refer to `SparseMethodSetup` in `config.hh`
            ctx->nz_density        = StrHelper::str2fp(v);
            ctx->nz_density_factor = 1 / ctx->nz_density;
        }
        else if (optmatch({"densityfactor"})) {  // refer to `SparseMethodSetup` in `config.hh`
            ctx->nz_density_factor = StrHelper::str2fp(v);
            ctx->nz_density        = 1 / ctx->nz_density_factor;
        }
        else if (optmatch({"gpuverify"}) and is_enabled(v)) {
            ctx->use.gpu_verify = true;
        }

        // when to enable anchor
        if (ctx->predictor == "spline3") {
            // unconditionally use anchor when it is spline3
            ctx->use.anchor = true;
        }
    }
}

void set_from_cli_input(cusz::context_t ctx, int const argc, char** const argv)
{
    int i = 1;

    auto check_next = [&]() {
        if (i + 1 >= argc) throw std::runtime_error("out-of-range at" + std::string(argv[i]));
    };

    std::string opt;
    auto optmatch = [&](std::vector<std::string> vs) -> bool { return ConfigHelper::check_opt_in_list(opt, vs); };

    while (i < argc) {
        if (argv[i][0] == '-') {
            opt = std::string(argv[i]);

            if (optmatch({"-c", "--config"})) {
                check_next();
                set_config(ctx, argv[++i]);
            }
            else if (optmatch({"-R", "--report"})) {
                check_next();
                set_report(ctx, argv[++i]);
            }
            else if (optmatch({"-h", "--help"})) {
                cusz::Context::print_doc(true);
                exit(0);
            }
            else if (optmatch({"-v", "--version"})) {
                std::cout << ">>>>  cusz build: " << cusz::VERSION_TEXT << "\n";
                exit(0);
            }
            else if (optmatch({"-m", "--mode"})) {
                check_next();
                ctx->mode = std::string(argv[++i]);
                if (ctx->mode == "r2r") ctx->preprocess.prescan = true;
            }
            else if (optmatch({"-e", "--eb", "--error-bound"})) {
                check_next();
                char* end;
                ctx->eb = std::strtod(argv[++i], &end);
            }
            else if (optmatch({"-p", "--predictor"})) {
                check_next();
                ctx->predictor = std::string(argv[++i]);
            }
            else if (optmatch({"-c", "--codec"})) {
                check_next();
                // placeholder
            }
            else if (optmatch({"-s", "--spcodec"})) {
                check_next();
                // placeholder
            }
            else if (optmatch({"-t", "--type", "--dtype"})) {
                check_next();
                std::string s = std::string(std::string(argv[++i]));
                if (s == "f32" or s == "fp4")
                    ctx->dtype = "f32";
                else if (s == "f64" or s == "fp8")
                    ctx->dtype = "f64";
            }
            else if (optmatch({"-i", "--input"})) {
                check_next();
                ctx->fname.fname = std::string(argv[++i]);
            }
            else if (optmatch({"-l", "--len"})) {
                check_next();
                cusz::Context::parse_input_length(argv[++i], ctx);
            }
            else if (optmatch({"-L", "--allocation-len"})) {
                check_next();
                // placeholder
            }
            else if (optmatch({"-z", "--zip", "--compress"})) {
                ctx->cli_task.construct = true;
            }
            else if (optmatch({"-x", "--unzip", "--decompress"})) {
                ctx->cli_task.reconstruct = true;
            }
            else if (optmatch({"-r", "--dry-run"})) {
                ctx->cli_task.dryrun = true;
            }
            else if (optmatch({"--anchor"})) {
                ctx->use.anchor = true;
            }
            else if (optmatch({"--nondestructive", "--input-nondestructive"})) {
                // placeholder
            }
            else if (optmatch({"--failfast"})) {
                // placeholder
            }
            else if (optmatch({"-P", "--pre", "--preprocess"})) {
                check_next();
                std::string pre(argv[++i]);
                if (pre.find("binning") != std::string::npos) { ctx->preprocess.binning = true; }
            }
            else if (optmatch({"-T", "--post", "--postprocess"})) {
                check_next();
                std::string post(argv[++i]);
                if (post.find("gzip") != std::string::npos) { ctx->postcompress.cpu_gzip = true; }
                if (post.find("nvcomp") != std::string::npos) { ctx->postcompress.gpu_nvcomp_cascade = true; }
            }
            else if (optmatch({"-V", "--verbose"})) {
                ctx->verbose = true;
            }
            else if (optmatch({"--pipeline"})) {
                check_next();
                ctx->pipeline = std::string(argv[++i]);
            }
            else if (optmatch({"--demo"})) {
                check_next();
                ctx->use.predefined_demo = true;
                ctx->demo_dataset        = std::string(argv[++i]);
                ctx->load_demo_sizes();
            }
            else if (optmatch({"-S", "-X", "--skip", "--exclude"})) {
                check_next();
                std::string exclude(argv[++i]);
                if (exclude.find("huffman") != std::string::npos) { ctx->skip.huffman = true; }
                if (exclude.find("write2disk") != std::string::npos) { ctx->skip.write2disk = true; }
            }
            else if (optmatch({"--opath"})) {
                check_next();
                ctx->opath = std::string(argv[++i]);
            }
            else if (optmatch({"--origin", "--compare"})) {
                check_next();
                ctx->fname.origin_cmp = std::string(argv[++i]);
            }
            else {
                const char* notif_prefix = "invalid option value at position ";
                char*       notif;
                int         size = asprintf(&notif, "%d: %s", i, argv[i]);
                cerr << LOG_ERR << notif_prefix << "\e[1m" << notif << "\e[0m"
                     << "\n";
                cerr << std::string(LOG_NULL.length() + strlen(notif_prefix), ' ');
                cerr << "\e[1m";
                cerr << std::string(strlen(notif), '~');
                cerr << "\e[0m\n";

                ctx->trap(-1);
            }
        }
        else {
            const char* notif_prefix = "invalid option at position ";
            char*       notif;
            int         size = asprintf(&notif, "%d: %s", i, argv[i]);
            cerr << LOG_ERR << notif_prefix << "\e[1m" << notif
                 << "\e[0m"
                    "\n"
                 << std::string(LOG_NULL.length() + strlen(notif_prefix), ' ')  //
                 << "\e[1m"                                                     //
                 << std::string(strlen(notif), '~')                             //
                 << "\e[0m\n";

            ctx->trap(-1);
        }
        i++;
    }
}

}  // namespace

cuszCTX& cuszCTX::set_control_string(const char* in_str)
{
    set_config(this, in_str);
    return *this;
}

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

void cuszCTX::validate()
{
    bool to_abort = false;
    if (fname.fname.empty()) {
        cerr << LOG_ERR << "must specify input file" << endl;
        to_abort = true;
    }

    if (data_len == 1 and not use.predefined_demo) {
        if (cli_task.construct or cli_task.dryrun) {
            cerr << LOG_ERR << "wrong input size" << endl;
            to_abort = true;
        }
    }
    if (not cli_task.construct and not cli_task.reconstruct and not cli_task.dryrun) {
        cerr << LOG_ERR << "select compress (-z), decompress (-x) or dry-run (-r)" << endl;
        to_abort = true;
    }
    if (false == ConfigHelper::check_dtype(dtype, false)) {
        if (cli_task.construct or cli_task.dryrun) {
            std::cout << dtype << endl;
            cerr << LOG_ERR << "must specify data type" << endl;
            to_abort = true;
        }
    }

    if (quant_bytewidth == 1)
        assert(dict_size <= 256);
    else if (quant_bytewidth == 2)
        assert(dict_size <= 65536);

    if (cli_task.dryrun and cli_task.construct and cli_task.reconstruct) {
        cerr << LOG_WARN << "no need to dry-run, compress and decompress at the same time" << endl;
        cerr << LOG_WARN << "dryrun only" << endl << endl;
        cli_task.construct   = false;
        cli_task.reconstruct = false;
    }
    else if (cli_task.dryrun and cli_task.construct) {
        cerr << LOG_WARN << "no need to dry-run and compress at the same time" << endl;
        cerr << LOG_WARN << "dryrun only" << endl << endl;
        cli_task.construct = false;
    }
    else if (cli_task.dryrun and cli_task.reconstruct) {
        cerr << LOG_WARN << "no need to dry-run and decompress at the same time" << endl;
        cerr << LOG_WARN << "will dryrun only" << endl << endl;
        cli_task.reconstruct = false;
    }

    if (to_abort) {
        print_doc();
        exit(-1);
    }
}

cuszCTX::cuszCTX(int argc, char** const argv)
{
    std::string opt;
    auto optmatch = [&](std::vector<std::string> vs) -> bool { return ConfigHelper::check_opt_in_list(opt, vs); };

    if (argc == 1) {
        print_doc();
        exit(0);
    }

    /******************************************************************************/
    /* phase 0: parse */
    set_from_cli_input(this, argc, argv);

    // special treatment
    if (predictor == "spline3") {
        // unconditionally use anchor when it is spline3
        use.anchor = true;
    }

    /******************************************************************************/
    /* phase 1: check syntax */
    if (read_args_status != 0) {
        std::cout << LOG_INFO << "Exiting..." << endl;
        // after printing ALL argument errors
        exit(-1);
    }

    /******************************************************************************/
    /* phase 2: check if legal */
    validate();

    /******************************************************************************/
    /* phase 3: sort out filenames */
    derive_fnames();
}

cuszCTX::cuszCTX(const char* in_str, bool dbg_print)
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

    set_config(this, in_str, dbg_print);
}

void cuszCTX::print_doc(bool full)
{
    std::cout << "\n>>>>  cusz build: " << cusz::VERSION_TEXT << "\n";

    if (full)
        std::cout << StrHelper::doc_format(cusz_full_doc) << std::endl;
    else
        std::cout << cusz_short_doc << std::endl;
}

void cuszCTX::derive_fnames()
{
    // (1) "fname"          -> "", "fname"
    // (2) "./fname"        -> "./" "fname"
    // (3) "/path/to/fname" -> "/path/to", "fname"
    auto input_path = fname.fname.substr(0, fname.fname.rfind('/') + 1);
    if (not cli_task.construct and cli_task.reconstruct) fname.fname = fname.fname.substr(0, fname.fname.rfind('.'));
    fname.basename = fname.fname.substr(fname.fname.rfind('/') + 1);

    if (opath.empty()) opath = input_path.empty() ? opath = "" : opath = input_path;
    opath += "/";

    fname.path_basename   = opath + fname.basename;
    fname.compress_output = fname.path_basename + ".cusza";
}
