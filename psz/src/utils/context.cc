/**
 * @file context.cc
 * @author Jiannan Tian
 * @brief context struct with argument parser
 * @version 0.1
 * @date 2020-09-20
 * Created on: 20-04-24
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory See LICENSE in top-level directory
 *
 */

#include "cusz/context.h"

#include <cxxabi.h>

#include <fstream>

#include "cusz/header.h"
#include "cusz/type.h"
#include "detail/busyheader.hh"
#include "document.inl"
#include "utils/format.hh"
#include "utils/verinfo.h"

using std::cerr;
using std::endl;
using ss_t = std::stringstream;
using map_t = std::unordered_map<std::string, std::string>;
using str_list = std::vector<std::string>;

namespace psz {

#if defined(PSZ_USE_CUDA)

const char* BACKEND_TEXT = "cuSZ";
const char* VERSION_TEXT = "2025-02-05 (0.16)";
const int VERSION = 20241218;

#elif defined(PSZ_USE_HIP)

const char* BACKEND_TEXT = "hipSZ";
const char* VERSION_TEXT = "2023-08-31 (unstable)";
const int VERSION = 20230831;

#elif defined(PSZ_USE_1API)

const char* BACKEND_TEXT = "dpSZ";
const char* VERSION_TEXT = "2023-09-28 (unstable)";
const int VERSION = 20230928;

#endif

const int COMPATIBILITY = 0;

}  // namespace psz

void capi_psz_version() { printf("\n>>> %s build: %s\n", psz::BACKEND_TEXT, psz::VERSION_TEXT); }

void capi_psz_versioninfo()
{
  capi_psz_version();
  printf("\ntoolchain:\n");
  print_CXX_ver();
  print_NVCC_ver();
  printf("\ndriver:\n");
  print_CUDA_driver();
  print_NVIDIA_driver();
  printf("\n");
  CUDA_devices();
}

namespace psz {

struct str_helper {
  static unsigned int str2int(const char* s);
  static unsigned int str2int(std::string s);
  static double str2fp(const char* s);
  static double str2fp(std::string s);
  static bool is_kv_pair(std::string s);
  static std::pair<std::string, std::string> separate_kv(std::string& s);
  static void parse_strlist_as_kv(const char* in_str, map_t& kv_list);
  static void parse_strlist(const char* in_str, str_list& list);
  static std::pair<std::string, bool> parse_kv_onoff(std::string in_str);
  static std::string doc_format(const std::string& s);
  static std::string nnz_percentage(uint32_t nnz, uint32_t data_len);
  static void check_cuszmode(const std::string& val);
  static bool check_dtype(const std::string& val, bool delay_failure = true);
  static bool check_dtype(const psz_dtype& val, bool delay_failure = true);
  static bool check_opt_in_list(std::string const& opt, std::vector<std::string> vs);
  static void parse_length_literal(const char* str, std::vector<std::string>& dims);
  static size_t filesize(std::string fname);
  template <typename T1, typename T2>
  static size_t get_npart(T1 size, T2 subsize);
  template <typename TRIO>
  static bool eq(TRIO a, TRIO b);
  static void print_datasegment_tablehead();
  static std::string demangle(const char* name);
  static void print_document(bool full_document);
  static void parse_argv(psz_ctx* ctx, int const argc, char** const argv);
  static void parse_length(psz_ctx* ctx, const char* lenstr);
  static void parse_length_zyx(psz_ctx* ctx, const char* lenstr);
  static void parse_control_string_Hi(psz_ctx* ctx, const char* in_str, bool dbg_print);
  static void validate_args(psz_ctx* ctx);
  static void set_report(psz_ctx* ctx, const char* in_str);
  static void set_datadump(psz_ctx* ctx, const char* in_str);
  static void set_radius(psz_ctx* ctx, int _);
  static void set_huffchunk(psz_ctx* ctx, int _);
  static void set_densityfactor(psz_ctx* ctx, int _);
};

}  // namespace psz

unsigned int psz::str_helper::str2int(const char* s)
{
  char* end;
  auto res = std::strtol(s, &end, 10);
  if (*end) {
    const char* notif = "invalid option value, non-convertible part: ";
    cerr << LOG_ERR << notif << "\e[1m" << s << "\e[0m" << endl;
  }
  return res;
}

unsigned int psz::str_helper::str2int(std::string s) { return str2int(s.c_str()); }

double psz::str_helper::str2fp(const char* s)
{
  char* end;
  auto res = std::strtod(s, &end);
  if (*end) {
    const char* notif = "invalid option value, non-convertible part: ";
    cerr << LOG_ERR << notif << "\e[1m" << end << "\e[0m" << endl;
  }
  return res;
}

double psz::str_helper::str2fp(std::string s) { return str2fp(s.c_str()); }

bool psz::str_helper::is_kv_pair(std::string s) { return s.find("=") != std::string::npos; }

std::pair<std::string, std::string> psz::str_helper::separate_kv(std::string& s)
{
  std::string delimiter = "=";

  if (s.find(delimiter) == std::string::npos)
    throw std::runtime_error("\e[1mnot a correct key-value syntax, must be \"opt=value\"\e[0m");

  std::string k = s.substr(0, s.find(delimiter));
  std::string v = s.substr(s.find(delimiter) + delimiter.length(), std::string::npos);

  return std::make_pair(k, v);
}

void psz::str_helper::parse_strlist_as_kv(const char* in_str, map_t& kv_list)
{
  ss_t ss(in_str);
  while (ss.good()) {
    std::string tmp;
    std::getline(ss, tmp, ',');
    kv_list.insert(separate_kv(tmp));
  }
}

void psz::str_helper::parse_strlist(const char* in_str, str_list& list)
{
  ss_t ss(in_str);
  while (ss.good()) {
    std::string tmp;
    std::getline(ss, tmp, ',');
    list.push_back(tmp);
  }
}

std::pair<std::string, bool> psz::str_helper::parse_kv_onoff(std::string in_str)
{
  auto kv_literal = "(.*?)=(on|ON|off|OFF)";
  std::regex kv_pattern(kv_literal);
  std::regex onoff_pattern("on|ON|off|OFF");

  bool onoff = false;
  std::string k, v;

  std::smatch kv_match;
  if (std::regex_match(in_str, kv_match, kv_pattern)) {
    // the 1st match: whole string
    // the 2nd: k, the 3rd: v
    if (kv_match.size() == 3) {
      k = kv_match[1].str(), v = kv_match[2].str();

      std::smatch v_match;
      if (std::regex_match(v, v_match, onoff_pattern)) {  //
        onoff = (v == "on") or (v == "ON");
      }
      else {
        throw std::runtime_error("not legal (k=v)-syntax");
      }
    }
  }
  return std::make_pair(k, onoff);
}

std::string psz::str_helper::doc_format(const std::string& s)
{
  std::regex gray("%(.*?)%");
  std::string gray_text("\e[37m$1\e[0m");

  std::regex bful("@(.*?)@");
  std::string bful_text("\e[1m\e[4m$1\e[0m");
  std::regex bf("\\*(.*?)\\*");
  std::string bf_text("\e[1m$1\e[0m");
  std::regex ul(R"(_((\w|-|\d|\.)+?)_)");
  std::string ul_text("\e[4m$1\e[0m");
  std::regex red(R"(\^\^(.*?)\^\^)");
  std::string red_text("\e[31m$1\e[0m");

  auto a = std::regex_replace(s, bful, bful_text);
  auto b = std::regex_replace(a, bf, bf_text);
  auto c = std::regex_replace(b, ul, ul_text);
  auto d = std::regex_replace(c, red, red_text);
  auto e = std::regex_replace(d, gray, gray_text);

  return e;
}

std::string psz::str_helper::nnz_percentage(uint32_t nnz, uint32_t data_len)
{
  return "(" + std::to_string(nnz / 1.0 / data_len * 100) + "%)";
}

void psz::str_helper::check_cuszmode(const std::string& val)
{
  auto legal = (val == "r2r" or val == "rel") or (val == "abs");
  if (not legal) throw std::runtime_error("`mode` must be \"r2r\" or \"abs\".");
}

bool psz::str_helper::check_dtype(const std::string& val, bool delay_failure)
{
  auto legal = (val == "f32") or (val == "f4") or (val == "f64") or (val == "f8");
  if (not legal)
    if (not delay_failure)
      throw std::runtime_error("Only `f32`/`f4` or `f64`/`f8` is supported temporarily.");

  return legal;
}

bool psz::str_helper::check_dtype(const psz_dtype& val, bool delay_failure)
{
  auto legal = (val == F4) or (val == F8);
  if (not legal)
    if (not delay_failure)
      throw std::runtime_error("Only `f32`/`f4` or `f64`/`f8` is supported temporarily.");

  return legal;
}

bool psz::str_helper::check_opt_in_list(std::string const& opt, std::vector<std::string> vs)
{
  for (auto& i : vs) {
    if (opt == i) return true;
  }
  return false;
}

void psz::str_helper::parse_length_literal(const char* str, std::vector<std::string>& dims)
{
  std::stringstream data_len_ss(str);
  auto data_len_literal = data_len_ss.str();
  auto checked = false;

  for (auto s : {"x", "*", "-", ",", "m"}) {
    if (checked) break;
    char delimiter = s[0];

    if (data_len_literal.find(delimiter) != std::string::npos) {
      while (data_len_ss.good()) {
        std::string substr;
        std::getline(data_len_ss, substr, delimiter);
        dims.push_back(substr);
      }
      checked = true;
    }
  }

  // handle 1D
  if (not checked) { dims.push_back(data_len_literal); }
}

size_t psz::str_helper::filesize(std::string fname)
{
  std::ifstream in(fname.c_str(), std::ifstream::ate | std::ifstream::binary);
  return in.tellg();
}

template <typename T1, typename T2>
size_t psz::str_helper::get_npart(T1 size, T2 subsize)
{
  static_assert(
      std::numeric_limits<T1>::is_integer and std::numeric_limits<T2>::is_integer,
      "[get_npart] must be plain interger types.");

  return (size + subsize - 1) / subsize;
}

template <typename TRIO>
bool psz::str_helper::eq(TRIO a, TRIO b)
{
  return (a.x == b.x) and (a.y == b.y) and (a.z == b.z);
}

void psz::str_helper::print_datasegment_tablehead()
{
  printf(
      "\ndata segments:\n  \e[1m\e[31m%-18s\t%12s\t%15s\t%15s\e[0m\n",  //
      const_cast<char*>("name"),                                        //
      const_cast<char*>("nbyte"),                                       //
      const_cast<char*>("start"),                                       //
      const_cast<char*>("end"));
}

std::string psz::str_helper::demangle(const char* name)
{
  int status = -4;
  char* res = abi::__cxa_demangle(name, nullptr, nullptr, &status);

  const char* const demangled_name = (status == 0) ? res : name;
  std::string ret_val(demangled_name);
  free(res);
  return ret_val;
}

void psz::str_helper::set_report(psz_ctx* ctx, const char* in_str)
{
  str_list opts;
  psz::str_helper::parse_strlist(in_str, opts);

  for (auto o : opts) {
    // printf("[psz::dbg::parse] opt: %s\n", o.c_str());
    if (psz::str_helper::is_kv_pair(o)) {
      auto kv = psz::str_helper::parse_kv_onoff(o);

      if (kv.first == "cr") ctx->cli->report_cr = kv.second;
      // else if (kv.first == "cr.est")
      //   ctx->cli->report_cr_est = kv.second;
      else if (kv.first == "time")
        ctx->cli->report_time = kv.second;
    }
    else {
      if (o == "cr") ctx->cli->report_cr = true;
      // else if (o == "cr.est")
      //   ctx->cli->report_cr_est = true;
      else if (o == "time")
        ctx->cli->report_time = true;
    }
  }
}

void psz::str_helper::set_datadump(psz_ctx* ctx, const char* in_str)
{
  str_list opts;
  psz::str_helper::parse_strlist(in_str, opts);

  for (auto o : opts) {
    if (psz::str_helper::is_kv_pair(o)) {
      auto kv = psz::str_helper::parse_kv_onoff(o);

      if (kv.first == "quantcode" or kv.first == "quant")
        ctx->cli->dump_quantcode = kv.second;
      else if (kv.first == "histogram" or kv.first == "hist")
        ctx->cli->dump_hist = kv.second;
      else if (kv.first == "full_huffman_binary" or kv.first == "full_hf")
        ctx->cli->dump_full_hf = kv.second;
    }
    else {
      if (o == "quantcode" or o == "quant")
        ctx->cli->dump_quantcode = true;
      else if (o == "histogram" or o == "hist")
        ctx->cli->dump_hist = true;
      else if (o == "full_huffman_binary" or o == "full_hf")
        ctx->cli->dump_full_hf = true;
    }
  }
}

// Extra configuration for the cuSZ-Hi variant.
// Syntax: comma-separated key-pairs: "key1=val1,key2=val2[,...]".
void psz::str_helper::parse_control_string_Hi(psz_ctx* ctx, const char* in_str, bool dbg_print)
{
  map_t opts;
  psz::str_helper::parse_strlist_as_kv(in_str, opts);

  if (dbg_print) {
    for (auto kv : opts) printf("%-*s %-s\n", 10, kv.first.c_str(), kv.second.c_str());
    std::cout << "\n";
  }

  std::string k, v;
  char* end;

  auto optmatch = [&](std::vector<std::string> vs) -> bool {
    return psz::str_helper::check_opt_in_list(k, vs);
  };
  auto is_enabled = [&](auto& v) -> bool { return v == "on" or v == "ON"; };

  for (auto kv : opts) {
    k = kv.first;
    v = kv.second;

    //// cuSZ-Hi configs
    if (optmatch({"auto_tuning", "auto-tuning"})) {
      if (v == "cr-first" or v == "CR-first")
        CLI_interp_params(ctx)->auto_tuning = 3;
      else if (v == "rd-first" or v == "RD-first")
        CLI_interp_params(ctx)->auto_tuning = 6;
      else {
        try {
          CLI_interp_params(ctx)->auto_tuning = static_cast<uint8_t>(psz::str_helper::str2int(v));
        }
        catch (...) {
          std::cerr << "[Error] Invalid `auto_tuning` value: " << v
                    << ". Expected cr-first, rd-first, or an integer.\n";
          exit(1);
        }
      }
    }
    else if (optmatch({"alpha", "intp-alpha"}))
      CLI_interp_params(ctx)->alpha = psz::str_helper::str2fp(v);
    else if (optmatch({"beta", "intp-beta"}))
      CLI_interp_params(ctx)->beta = psz::str_helper::str2fp(v);
    else if (optmatch({"md_0", "md0"}))
      CLI_interp_params(ctx)->use_md[0] = psz::str_helper::str2int(v);
    else if (optmatch({"md_1", "md1"}))
      CLI_interp_params(ctx)->use_md[1] = psz::str_helper::str2int(v);
    else if (optmatch({"md_2", "md2"}))
      CLI_interp_params(ctx)->use_md[2] = psz::str_helper::str2int(v);
    else if (optmatch({"md_3", "md3"}))
      CLI_interp_params(ctx)->use_md[3] = psz::str_helper::str2int(v);
    else if (optmatch({"nat_0", "nat0"}))
      CLI_interp_params(ctx)->use_natural[0] = psz::str_helper::str2int(v);
    else if (optmatch({"nat_1", "nat1"}))
      CLI_interp_params(ctx)->use_natural[1] = psz::str_helper::str2int(v);
    else if (optmatch({"nat_2", "nat2"}))
      CLI_interp_params(ctx)->use_natural[2] = psz::str_helper::str2int(v);
    else if (optmatch({"nat_3", "nat3"}))
      CLI_interp_params(ctx)->use_natural[3] = psz::str_helper::str2int(v);
    else if (optmatch({"rev_0", "rev0"}))
      CLI_interp_params(ctx)->reverse[0] = psz::str_helper::str2int(v);
    else if (optmatch({"rev_1", "rev1"}))
      CLI_interp_params(ctx)->reverse[1] = psz::str_helper::str2int(v);
    else if (optmatch({"rev_2", "rev2"}))
      CLI_interp_params(ctx)->reverse[2] = psz::str_helper::str2int(v);
    else if (optmatch({"rev_3", "rev3"}))
      CLI_interp_params(ctx)->reverse[3] = psz::str_helper::str2int(v);
  }
}

void pszctx_create_from_argv(psz_ctx* ctx, int const argc, char** const argv)
{
  if (argc == 1) {
    psz::str_helper::print_document(false);
    exit(0);
  }

  psz::str_helper::parse_argv(ctx, argc, argv);
  psz::str_helper::validate_args(ctx);
}

void psz::str_helper::parse_argv(psz_ctx* ctx, int const argc, char** const argv)
{
  int i = 1;

  auto check_next = [&]() {
    if (i + 1 >= argc) throw std::runtime_error("out-of-range at" + std::string(argv[i]));
  };

  std::string opt;
  auto optmatch = [&](std::vector<std::string> vs) -> bool {
    return psz::str_helper::check_opt_in_list(opt, vs);
  };

  while (i < argc) {
    if (argv[i][0] == '-') {
      opt = std::string(argv[i]);

      if (optmatch({"-c", "--config", "--hi-config"})) {
        check_next();
        psz::str_helper::parse_control_string_Hi(ctx, argv[++i], false);
      }
      else if (optmatch({"-R", "--report"})) {
        check_next();
        psz::str_helper::set_report(ctx, argv[++i]);
      }
      else if (optmatch({"--dump"})) {
        check_next();
        psz::str_helper::set_datadump(ctx, argv[++i]);
      }
      else if (optmatch({"-h", "--help"})) {
        psz::str_helper::print_document(true);
        exit(0);
      }
      else if (optmatch({"-v", "--version"})) {
        capi_psz_version();
        exit(0);
      }
      else if (optmatch({"-V", "--versioninfo", "--query-env"})) {
        capi_psz_versioninfo();
        exit(0);
      }
      else if (optmatch({"-m", "--mode"})) {
        check_next();
        auto _ = std::string(argv[++i]);
        ctx->header->mode = (_ == "r2r" or _ == "rel") ? Rel : Abs;
        if (ctx->header->mode == Rel) ctx->cli->rel_range_scan = true;
        strcpy(ctx->cli->char_mode, _.c_str());
      }
      else if (optmatch({"-e", "--eb", "--error-bound"})) {
        check_next();
        char* end;
        ctx->header->eb = std::strtod(argv[++i], &end);
        strcpy(ctx->cli->char_meta_eb, argv[i]);
      }
      else if (optmatch({"-p", "--pred", "--predictor"})) {
        check_next();
        auto v = std::string(argv[++i]);
        strcpy(ctx->cli->char_predictor_name, v.c_str());

        if (v == "spline" or v == "spline3" or v == "spl")
          ctx->header->pred_type = psz_predtype::Spline;
        else if (v == "lorenzo" or v == "lrz")
          ctx->header->pred_type = psz_predtype::Lorenzo;
        else if (v == "lorenzo-zigzag" or v == "lrz-zz")
          ctx->header->pred_type = psz_predtype::LorenzoZigZag;
        else if (v == "lorenzo-proto" or v == "lrz-proto")
          ctx->header->pred_type = psz_predtype::LorenzoProto;
        else
          printf(
              "[psz::warning::parser] "
              "\"%s\" is not a supported predictor; "
              "fallback to \"lorenzo\".",
              v.c_str());
      }
      else if (optmatch({"--hist", "--histogram"})) {
        check_next();
        auto v = std::string(argv[++i]);
        strcpy(ctx->cli->char_predictor_name, v.c_str());

        if (v == "generic")
          ctx->header->hist_type = psz_histotype::HistogramGeneric;
        else if (v == "sparse")
          ctx->header->hist_type = psz_histotype::HistogramSparse;
      }
      else if (optmatch({"-c1", "--codec", "--codec1"})) {
        check_next();
        auto v = std::string(argv[++i]);
        strcpy(ctx->cli->char_codec1_name, v.c_str());

        if (v == "huffman" or v == "hf")
          ctx->header->codec1_type = psz_codectype::Huffman;
        else if (v == "fzgcodec")
          ctx->header->codec1_type = psz_codectype::FZGPUCodec;
      }
      else if (optmatch({"-t", "--type", "--dtype"})) {
        check_next();
        std::string s = std::string(std::string(argv[++i]));
        if (s == "f32" or s == "f4")
          ctx->header->dtype = F4;
        else if (s == "f64" or s == "f8")
          ctx->header->dtype = F8;
      }
      else if (optmatch({"-i", "--input"})) {
        check_next();
        auto _ = std::string(argv[++i]);
        strcpy(ctx->cli->file_input, _.c_str());
      }
      else if (optmatch({"-l", "--len", "--xyz", "--dim3"})) {
        check_next();
        psz::str_helper::parse_length(ctx, argv[++i]);
      }
      else if (optmatch({"--math-order", "--zyx", "--slowest-to-fastest"})) {
        check_next();
        psz::str_helper::parse_length_zyx(ctx, argv[++i]);
      }
      else if (optmatch({"-z", "--zip", "--compress"})) {
        ctx->cli->task_construct = true;
      }
      else if (optmatch({"-x", "--unzip", "--decompress"})) {
        ctx->cli->task_reconstruct = true;
      }
      else if (optmatch({"-V", "--verbose"})) {
        ctx->cli->verbose = true;
      }
      else if (optmatch({"-S", "-X", "--skip", "--exclude"})) {
        check_next();
        std::string exclude(argv[++i]);
        if (exclude.find("huffman") != std::string::npos) { ctx->cli->skip_hf = true; }
        if (exclude.find("write2disk") != std::string::npos) { ctx->cli->skip_tofile = true; }
      }
      else if (optmatch({"--opath"})) {
        check_next();
        throw std::runtime_error("[23june] Specifying output path is temporarily disabled.");
        auto _ = std::string(argv[++i]);
        strcpy(ctx->cli->opath, _.c_str());
      }
      else if (optmatch({"--origin", "--compare"})) {
        check_next();
        auto _ = std::string(argv[++i]);
        strcpy(ctx->cli->file_compare, _.c_str());
      }
      else if (optmatch({"-a", "--auto"})) {
        check_next();
        std::string at_mode = argv[++i];
        if (at_mode == "cr-first" or at_mode == "CR-first")
          CLI_interp_params(ctx)->auto_tuning = 3;
        else if (at_mode == "rd-first" or at_mode == "RD-first")
          CLI_interp_params(ctx)->auto_tuning = 6;
        else {
          try {
            CLI_interp_params(ctx)->auto_tuning = static_cast<uint8_t>(std::stoi(at_mode));
          }
          catch (...) {
            std::cerr << "[Error] Unknown auto-tuning mode: " << at_mode
                      << ". Supported: cr-first, rd-first, or an integer value." << std::endl;
            exit(1);
          }
        }
      }
      else if (optmatch({"-s", "--scheme"})) {
        check_next();
        auto _ = std::string(argv[++i]);
        if (_ == "tp" or _ == "TP" or _ == "speed") { ctx->header->codec1_type = LC; }
        else if (_ == "cr" or _ == "CR") {
          ctx->header->codec1_type = Huffman;
        }
      }
      else if (optmatch({"--sycl-device"})) {
#if defined(PSZ_USE_1API)
        check_next();
        auto _v = string(argv[++i]);
        if (_v == "cpu" or _v == "CPU")
          ctx->device = CPU;
        else if (_v == "gpu" or _v == "GPU")
          ctx->device = INTELGPU;
        else
          ctx->device = INTELGPU;

#else
        throw std::runtime_error(
            "[psz::error] --sycl-device is not supported backend other than "
            "CUDA/HIP.");
#endif
      }
      else {
        const char* notif_prefix = "invalid option value at position ";
        char* notif;
        int size = asprintf(&notif, "%d: %s", i, argv[i]);
        cerr << LOG_ERR << notif_prefix << "\e[1m" << notif << "\e[0m" << "\n";
        cerr << std::string(strlen(LOG_NULL) + strlen(notif_prefix), ' ');
        cerr << "\e[1m";
        cerr << std::string(strlen(notif), '~');
        cerr << "\e[0m\n";

        std::cout << LOG_ERR << "Exiting..." << endl;
        exit(-1);
      }
    }
    else {
      const char* notif_prefix = "invalid option at position ";
      char* notif;
      int size = asprintf(&notif, "%d: %s", i, argv[i]);
      cerr << LOG_ERR << notif_prefix << "\e[1m" << notif
           << "\e[0m"
              "\n"
           << std::string(strlen(LOG_NULL) + strlen(notif_prefix), ' ')  //
           << "\e[1m"                                                    //
           << std::string(strlen(notif), '~')                            //
           << "\e[0m\n";

      std::cout << LOG_ERR << "Exiting..." << endl;
      exit(-1);
    }
    i++;
  }
}

void psz::str_helper::parse_length(psz_ctx* ctx, const char* lenstr)
{
  std::vector<std::string> dims;
  psz::str_helper::parse_length_literal(lenstr, dims);
  ctx->ndim = dims.size();
  ctx->header->y = ctx->header->z = ctx->header->w = 1;
  ctx->header->x = psz::str_helper::str2int(dims[0]);
  if (ctx->ndim >= 2) ctx->header->y = psz::str_helper::str2int(dims[1]);
  if (ctx->ndim >= 3) ctx->header->z = psz::str_helper::str2int(dims[2]);
  if (ctx->ndim >= 4) ctx->header->w = psz::str_helper::str2int(dims[3]);
  ctx->data_len = ctx->header->x * ctx->header->y * ctx->header->z * ctx->header->w;
}

void psz::str_helper::parse_length_zyx(psz_ctx* ctx, const char* lenstr)
{
  std::vector<std::string> dims;
  psz::str_helper::parse_length_literal(lenstr, dims);
  ctx->ndim = dims.size();
  ctx->header->y = ctx->header->z = ctx->header->w = 1;
  ctx->header->x = psz::str_helper::str2int(dims[ctx->ndim - 1]);
  if (ctx->ndim >= 2) ctx->header->y = psz::str_helper::str2int(dims[ctx->ndim - 2]);
  if (ctx->ndim >= 3) ctx->header->z = psz::str_helper::str2int(dims[ctx->ndim - 3]);
  if (ctx->ndim >= 4) ctx->header->w = psz::str_helper::str2int(dims[ctx->ndim - 4]);
  ctx->data_len = ctx->header->x * ctx->header->y * ctx->header->z * ctx->header->w;
}

void psz::str_helper::validate_args(psz_ctx* ctx)
{
  bool to_abort = false;
  // if (ctx->cli->file_input.empty()) {
  if (ctx->cli->file_input[0] == '\0') {
    cerr << LOG_ERR << "must specify input file" << endl;
    to_abort = true;
  }

  if (not ctx->cli->task_construct and not ctx->cli->task_reconstruct) {
    cerr << LOG_ERR << "select compress (-z) or decompress (-x)." << endl;
    to_abort = true;
  }
  if (false == psz::str_helper::check_dtype(ctx->header->dtype)) {
    if (ctx->cli->task_construct) {
      std::cout << ctx->header->dtype << endl;
      cerr << LOG_ERR << "must specify data type" << endl;
      to_abort = true;
    }
  }

  if (to_abort) {
    psz::str_helper::print_document(false);
    exit(-1);
  }
}

void psz::str_helper::print_document(bool full_document)
{
  if (full_document) {
    capi_psz_version();
    std::cout << "\n" << psz::str_helper::doc_format(psz_full_doc);
  }
  else {
    capi_psz_version();
    std::cout << psz::str_helper::doc_format(psz_short_doc);
  }
}

void pszctx_set_rawlen(psz_ctx* ctx, size_t _x, size_t _y, size_t _z)
{
  ctx->header->x = _x, ctx->header->y = _y, ctx->header->z = _z;

  auto ndim = 4;
  if (ctx->header->w == 1) ctx->ndim = 3;
  if (ctx->header->z == 1) ndim = 2;
  if (ctx->header->y == 1) ndim = 1;

  ctx->ndim = ndim;
  ctx->data_len = ctx->header->x * ctx->header->y * ctx->header->z;

  if (ctx->data_len == 1) throw std::runtime_error("Input data length cannot be 1 (linearized).");
  if (ctx->data_len == 0) throw std::runtime_error("Input data length cannot be 0 (linearized).");
}

void pszctx_set_len(psz_ctx* ctx, psz_len3 len) { pszctx_set_rawlen(ctx, len.x, len.y, len.z); }

psz_len3 pszctx_get_len3(psz_ctx* ctx)
{
  return psz_len3{ctx->header->x, ctx->header->y, ctx->header->z};
}

void psz::str_helper::set_radius(psz_ctx* ctx, int _)
{
  ctx->header->radius = _;
  ctx->dict_size = ctx->header->radius * 2;
}

psz_ctx* pszctx_default_values()
{
  return new psz_ctx{
      .header =
          new psz_header{
              .dtype = F4,
              .pred_type = DEFAULT_PREDICTOR,
              .hist_type = DEFAULT_HISTOGRAM,
              .codec1_type = DEFAULT_CODEC,
              .mode = Rel,
              .eb = 0.1,
              .radius = 512,
              .vle_sublen = 512,
              .vle_pardeg = -1,
              .x = 1,
              .y = 1,
              .z = 1,
              .w = 1,
              .splen = 0,
              .intp_param = make_default_params(),
          },
      .cli =
          new psz_cli_config{
              .dump_quantcode = false,
              .dump_hist = false,
              .task_construct = false,
              .task_reconstruct = false,
              .rel_range_scan = false,
              .use_gpu_verify = false,
              .skip_tofile = false,
              .skip_hf = false,
              .report_time = false,
              .report_cr = false,
              .verbose = false,
          },
      .dict_size = 1024,
      .data_len = 1,
      .ndim = -1,
      .there_is_memerr = false,
  };
}

void pszctx_set_default_values(psz_ctx* empty_ctx)
{
  auto default_vals = pszctx_default_values();
  memcpy(empty_ctx, default_vals, sizeof(psz_ctx));
  delete default_vals;
}

psz_ctx* pszctx_minimal_workset(
    psz_dtype const dtype, psz_predtype const predictor, int const quantizer_radius,
    psz_codectype const codec)
{
  auto ws = pszctx_default_values();
  ws->header->dtype = dtype;
  ws->header->pred_type = predictor;
  ws->header->codec1_type = codec;
  ws->dict_size = quantizer_radius * 2;
  ws->header->radius = quantizer_radius;
  return ws;
}

unsigned int CLI_x(psz_args* args) { return args->header->x; }
unsigned int CLI_y(psz_args* args) { return args->header->y; }
unsigned int CLI_z(psz_args* args) { return args->header->z; }
unsigned int CLI_w(psz_args* args) { return args->header->w; }
unsigned short CLI_radius(psz_args* args) { return args->header->radius; }
unsigned short CLI_bklen(psz_args* args) { return args->header->radius * 2; }
psz_dtype CLI_dtype(psz_args* args) { return args->header->dtype; }
psz_predtype CLI_predictor(psz_args* args) { return args->header->pred_type; }
psz_histotype CLI_hist(psz_args* args) { return args->header->hist_type; }
psz_codectype CLI_codec1(psz_args* args) { return args->header->codec1_type; }
psz_codectype CLI_codec2(psz_args* args) { return args->header->_future_codec2_type; }
psz_mode CLI_mode(psz_args* args) { return args->header->mode; }
double CLI_eb(psz_args* args) { return args->header->eb; }
psz_interp_params* CLI_interp_params(psz_ctx* ctx) { return &ctx->header->intp_param; };
