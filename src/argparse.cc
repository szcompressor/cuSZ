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

#include "argparse.hh"
#include "argument_parser/document.hh"
#include "utils/format.hh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

// TODO check version
const char* version_text  = "2021-09-08.2";
const int   version       = 202107132;
const int   compatibility = 0;

namespace {

unsigned int str2int(std::string s)
{
    char* end;
    auto  res = std::strtol(s.c_str(), &end, 10);
    if (*end) {
        const char* notif = "invalid option value, non-convertible part: ";
        cerr << log_err << notif << "\e[1m" << s << "\e[0m" << endl;
    }
    return res;
};

unsigned int str2int(const char* s)
{
    char* end;
    auto  res = std::strtol(s, &end, 10);
    if (*end) {
        const char* notif = "invalid option value, non-convertible part: ";
        cerr << log_err << notif << "\e[1m" << s << "\e[0m" << endl;
    }
    return res;
};

double str2fp(std::string s)
{
    char* end;
    auto  res = std::strtod(s.c_str(), &end);
    if (*end) {
        const char* notif = "invalid option value, non-convertible part: ";
        cerr << log_err << notif << "\e[1m" << end << "\e[0m" << endl;
    }
    return res;
}

double str2fp(const char* s)
{
    char* end;
    auto  res = std::strtod(s, &end);
    if (*end) {
        const char* notif = "invalid option value, non-convertible part: ";
        cerr << log_err << notif << "\e[1m" << end << "\e[0m" << endl;
    }
    return res;
};

std::pair<std::string, std::string> separate_kv(std::string& s)
{
    std::string delimiter = "=";

    if (s.find(delimiter) == std::string::npos)
        throw std::runtime_error("\e[1mnot a correct key-value syntax, must be \"opt=value\"\e[0m");

    std::string k = s.substr(0, s.find(delimiter));
    std::string v = s.substr(s.find(delimiter) + delimiter.length(), std::string::npos);

    return std::make_pair(k, v);
}

using ss_t     = std::stringstream;
using map_t    = std::unordered_map<std::string, std::string>;
using str_list = std::vector<std::string>;

auto parse_strlist_as_kv = [](char* in_str, map_t& kv_list) {
    ss_t ss(in_str);
    while (ss.good()) {
        std::string tmp;
        std::getline(ss, tmp, ',');
        kv_list.insert(separate_kv(tmp));
    }
};

auto parse_strlist = [](const char* in_str, str_list& list) {
    ss_t ss(in_str);
    while (ss.good()) {
        std::string tmp;
        std::getline(ss, tmp, ',');
        list.push_back(tmp);
    }
};

void set_preprocess(argpack* ap, const char* in_str)
{
    str_list opts;
    parse_strlist(in_str, opts);

    for (auto k : opts) {
        // TODO
    }
}

std::pair<std::string, bool> parse_kv_onoff(std::string in_str)
{
    auto       kv_literal = "(.*?)=(on|ON|off|OFF)";
    std::regex kv_pattern(kv_literal);
    std::regex onoff_pattern("on|ON|off|OFF");

    bool        onoff = false;
    std::string k, v;

    std::smatch kv_match;
    if (std::regex_match(in_str, kv_match, kv_pattern)) {
        // the 1st match: whole string
        // the 2nd: k, the 3rd: v
        if (kv_match.size() == 3) {
            k = kv_match[1].str();
            v = kv_match[2].str();

            std::smatch v_match;
            if (std::regex_match(v, v_match, onoff_pattern)) {  //
                onoff = (v == "on") or (v == "ON");
            }
            else {
                throw std::runtime_error("not (k=v)-syntax");
            }
        }
    }

    return std::make_pair(k, onoff);
}

void set_report(argpack* ap, const char* in_str)
{
    auto is_kv_pair = [](std::string s) { return s.find("=") != std::string::npos; };

    str_list opts;
    parse_strlist(in_str, opts);

    for (auto o : opts) {
        if (is_kv_pair(o)) {
            auto kv = parse_kv_onoff(o);

            // clang-format off
            if (kv.first == "quality")      ap->report.quality = kv.second;
            else if (kv.first == "cr")      ap->report.cr = kv.second;
            else if (kv.first == "compressibility") ap->report.compressibility = kv.second;
            else if (kv.first == "time")    ap->report.time = kv.second;
            // clang-format on
        }
        else {
            // clang-format off
            if (o == "quality")             ap->report.quality = true;
            else if (o == "cr")             ap->report.cr = true;
            else if (o == "compressibility")ap->report.compressibility = true;
            else if (o == "time")           ap->report.time = true;
            // clang-format on
        }
    }
}

void set_config(argpack* ap, char* in_str)
{
    map_t opts;
    parse_strlist_as_kv(in_str, opts);

    for (auto kv : opts) {
        // clang-format off
        if (kv.first == "mode")         { ap->mode = std::string(kv.second); }
        else if (kv.first == "eb")      { ap->eb = str2fp(kv.second); }
        else if (kv.first == "cap")     { ap->dict_size = str2int(kv.second), ap->radius = ap->dict_size / 2; }
        else if (kv.first == "huffbyte"){ ap->huff_nbyte = str2int(kv.second); }
        else if (kv.first == "quantbyte"){ ap->quant_nbyte = str2int(kv.second); }
        else if (kv.first == "quantbyte"){ ap->huffman_chunk = str2int(kv.second), ap->task_is.autotune_huffchunk = false; }
        else if (kv.first == "demo")    { ap->task_is.use_demo_dataset = true, ap->demo_dataset = string(kv.second); ap->load_demo_sizes();
        }
        // clang-format on
    }
}

}  // namespace

void ArgPack::load_demo_sizes()
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

string  //
ArgPack::format(const string& s)
{
    std::regex  gray("%(.*?)%");
    std::string gray_text("\e[37m$1\e[0m");

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
    auto        e = std::regex_replace(d, gray, gray_text);
    return e;
}

int  //
ArgPack::trap(int _status)
{
    this->read_args_status = _status;
    return read_args_status;
}

void  //
ArgPack::check_args()
{
    bool to_abort = false;
    if (fnames.path2file.empty()) {
        cerr << log_err << "Not specifying input file!" << endl;
        to_abort = true;
    }

    if (self_multiply4() == 1 and not task_is.use_demo_dataset) {
        if (task_is.construct or task_is.dryrun) {
            cerr << log_err << "Wrong input size(s)!" << endl;
            to_abort = true;
        }
    }
    if (not task_is.construct and not task_is.reconstruct and not task_is.dryrun) {
        cerr << log_err << "Select compress (-a), decompress (-x) or dry-run (-r)!" << endl;
        to_abort = true;
    }
    if (dtype != "f32" and dtype != "f64") {
        if (task_is.construct or task_is.dryrun) {
            cout << dtype << endl;
            cerr << log_err << "Not specifying data type!" << endl;
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
        cerr << log_warn << "No need to dry-run, compress and decompress at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        task_is.construct   = false;
        task_is.reconstruct = false;
    }
    else if (task_is.dryrun and task_is.construct) {
        cerr << log_warn << "No need to dry-run and compress at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
        task_is.construct = false;
    }
    else if (task_is.dryrun and task_is.reconstruct) {
        cerr << log_warn << "No need to dry-run and decompress at the same time!" << endl;
        cerr << log_warn << "Will dry run only." << endl << endl;
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
        print_cusz_short_doc();
        exit(-1);
    }
}

void  //
ArgPack::print_cusz_short_doc()
{
    cout << "\n>>>>  cusz build: " << version_text << "\n";
    cout << cusz_short_doc << endl;
}

void  //
ArgPack::print_cusz_full_doc()
{
    cout << "\n>>>>  cusz build: " << version_text << "\n";
    cout << format(cusz_full_doc) << endl;
}

void ArgPack::parse_args(int argc, char** argv)
{
    if (argc == 1) {
        print_cusz_short_doc();
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
                        std::stringstream   datalen(argv[++i]);
                        std::vector<string> dims;
                        while (datalen.good()) {
                            string substr;
                            getline(datalen, substr, ',');
                            dims.push_back(substr);
                        }
                        ndim = dims.size();
                        x = y = z = w = 1;
                        if (ndim == 1) {  //
                            x = str2int(dims[0].c_str());
                        }
                        if (ndim == 2) {  //
                            x = str2int(dims[0].c_str());
                            y = str2int(dims[1].c_str());
                        }
                        if (ndim == 3) {
                            x = str2int(dims[0].c_str());
                            y = str2int(dims[1].c_str());
                            z = str2int(dims[2].c_str());
                        }
                        if (ndim == 4) {
                            x = str2int(dims[0].c_str());
                            y = str2int(dims[1].c_str());
                            z = str2int(dims[2].c_str());
                            w = str2int(dims[3].c_str());
                        }
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
                    print_cusz_full_doc();
                    exit(0);
                case 'v':
                tag_version:
                    cout << ">>>>  cusz build: " << version_text << "\n";
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
    check_args();
    // phase 3: sort out filenames
    sort_out_fnames();
}

void ArgPack::sort_out_fnames()
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
