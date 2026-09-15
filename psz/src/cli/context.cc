// Author: Jiannan Tian
// context struct with argument parser

#include "pipeline.h"
#include "cusz/context.h"

#include <cstring>
#include <stdexcept>
#include <vector>

#include "arg_builder.hh"
#include "cli/document.inl"
#include "cli/verinfo.h"
#include "cusz/header.h"
#include "cusz/type.h"
#include "detail/check.hh"
#include "detail/kv_parse.hh"
#include "detail/str2num.hh"
#include "kv_binder.hh"
#include "utils/busyheader.hh"
#include "utils/demangle.hh"
#include "utils/format.hh"

using std::cerr;
using std::endl;
using std::string;

namespace psz {

#if defined(PSZ_USE_CUDA)

const char* BACKEND_TEXT = "cuSZ";
const char* VERSION_TEXT = "2025-02-05 (0.16)";
const int   VERSION      = 20241218;

#elif defined(PSZ_USE_1API)

const char* BACKEND_TEXT = "dpSZ";
const char* VERSION_TEXT = "2023-09-28 (unstable)";
const int   VERSION      = 20230928;

#endif

const int COMPATIBILITY = 0;

}  // namespace psz

void psz_version() { printf("\n>>> %s build: %s\n", psz::BACKEND_TEXT, psz::VERSION_TEXT); }

void psz_versioninfo()
{
  psz_version();
  printf("\ntoolchain:\n");
  print_CXX_ver();
  print_NVCC_ver();
  printf("\ndriver:\n");
  print_CUDA_driver();
  print_NVIDIA_driver();
  printf("\n");
  CUDA_devices();
}

// ---------------------------------------------------------------------------
// Bounded string copy: writes src into a fixed-size char[N] buffer; throws
// (rather than overflow) if src is too long.  N is inferred from the array.
// ---------------------------------------------------------------------------

template <size_t N>
static void apply_str(const string& src, char (&dst)[N])
{
  if (src.empty()) return;
  if (src.size() >= N)
    throw std::runtime_error("value '" + src + "' exceeds destination buffer size " +
                             std::to_string(N));
  std::memcpy(dst, src.c_str(), src.size() + 1);
}

static const auto report_binder = _ptb::kv_binder<psz_cli_config>()
                                      .flag({"cr"}, &psz_cli_config::report_cr)
                                      .flag({"time"}, &psz_cli_config::report_time);

// clang-format off
static const auto dump_binder = _ptb::kv_binder<psz_cli_config>()
  .flag({"quantcode", "quant"},             &psz_cli_config::dump_quantcode)
  .flag({"histogram", "hist"},              &psz_cli_config::dump_hist)
  .flag({"full_huffman_binary", "full_hf"}, &psz_cli_config::dump_full_hf);
// clang-format on

// clang-format off
static const auto hi_binder = _ptb::kv_binder<psz_interp_params>()
  .number({"alpha", "intp-alpha"}, &psz_interp_params::alpha)
  .number({"beta",  "intp-beta"},  &psz_interp_params::beta)
  .flag_ref({"md_0","md0"},  [](psz_interp_params& p) -> bool& { return p.use_md[0]; })
  .flag_ref({"md_1","md1"},  [](psz_interp_params& p) -> bool& { return p.use_md[1]; })
  .flag_ref({"md_2","md2"},  [](psz_interp_params& p) -> bool& { return p.use_md[2]; })
  .flag_ref({"md_3","md3"},  [](psz_interp_params& p) -> bool& { return p.use_md[3]; })
  .flag_ref({"nat_0","nat0"},[](psz_interp_params& p) -> bool& { return p.use_natural[0]; })
  .flag_ref({"nat_1","nat1"},[](psz_interp_params& p) -> bool& { return p.use_natural[1]; })
  .flag_ref({"nat_2","nat2"},[](psz_interp_params& p) -> bool& { return p.use_natural[2]; })
  .flag_ref({"nat_3","nat3"},[](psz_interp_params& p) -> bool& { return p.use_natural[3]; })
  .flag_ref({"rev_0","rev0"},[](psz_interp_params& p) -> bool& { return p.reverse[0]; })
  .flag_ref({"rev_1","rev1"},[](psz_interp_params& p) -> bool& { return p.reverse[1]; })
  .flag_ref({"rev_2","rev2"},[](psz_interp_params& p) -> bool& { return p.reverse[2]; })
  .flag_ref({"rev_3","rev3"},[](psz_interp_params& p) -> bool& { return p.reverse[3]; })
  .custom({"auto_tuning","auto-tuning"}, [](psz_interp_params& p, const string& _v) {
    if      (_v == "cr-first" or _v == "CR-first") p.auto_tuning = 3;
    else if (_v == "rd-first" or _v == "RD-first") p.auto_tuning = 6;
    else {
      auto n = _ptb::detail::str_to_int(_v.c_str());
      if (not n) throw std::runtime_error("invalid auto_tuning value: " + _v);
      p.auto_tuning = static_cast<uint8_t>(*n);
    }
  });
// clang-format on

// ---------------------------------------------------------------------------
// CLI schema — declared once, shared across all parse calls.
// ---------------------------------------------------------------------------

// clang-format off
static const auto psz_cli = _ptb::arg_builder("cusz")
  .string("input",    {"-i", "--input"},                              "",     "input file")
  .string("len",      {"-l", "--len", "--xyz", "--dim3"},             "",     "data dimensions (x,y,z order)")
  .string("len_zyx",  {"--math-order", "--zyx", "--slowest-to-fastest"}, "",  "data dimensions (z,y,x order)")
  .string("dtype",    {"-t", "--type", "--dtype"},                    "",     "f32/f4 or f64/f8")
  .string("eb",       {"-e", "--eb", "--error-bound"},                "0.1",  "error bound")
  .string("mode",     {"-m", "--mode"},                               "r2r",  "r2r (relative) or abs")
  .string("pred",     {"-p", "--pred", "--predictor"},                "",     "predictor")
  .string("hist",     {"--hist", "--histogram"},                      "",     "histogram type")
  .string("codec1",   {"-c1", "--codec", "--codec1"},                 "",     "primary codec")
  .string("codec2",   {"-c2", "--codec2"},                            "",     "secondary codec")
  .string("config",   {"--hi-config"},                                "",     "Hi-mode config key=val pairs")
  .string("report",   {"-R", "--report"},                             "",     "report options")
  .string("dump",     {"--dump"},                                     "",     "dump options")
  .string("skip",     {"-S", "-X", "--skip", "--exclude"},            "",     "skip: huffman, write2disk")
  .string("compare",  {"--origin", "--compare"},                      "",     "reference file for comparison")
  .string("auto",     {"-a", "--auto"},                               "",     "auto-tuning: cr-first, rd-first, int")
  .string("scheme",   {"-s", "--scheme"},                             "",     "shorthand: tp|speed or cr")
  .string("preset",   {"--preset"},                                   "",     "whole pipeline by name: fzg|hicr|hitp|hitp_r1")
  .string("pipeline", {"--pipeline"},                                 "",     "whole pipeline by stage: p1,c1[,c2]")
  .string("rmerge_count", {"--rmerge-count"},  "",   "HFR reduce-merge pass count 2|3|4; default is per codec")
  .flag("compress",   {"-z", "--zip", "--compress"},                          "run compression")
  .flag("decompress", {"-x", "--unzip", "--decompress"},                      "run decompression")
  .flag("verbose",    {"--verbose"},                                          "verbose output")
  .flag("hfd26",      {"--hfd26"},                "decode HFR-family archives with HFD26 (the default; stating it is a no-op)")
  .flag("hfd_coarse",  {"--hfd-coarse"},            "force the coarse one-thread-per-chunk decoder (HFR_coarse); HF and HF-rev2 are always coarse")
  ;
// clang-format on

// ---------------------------------------------------------------------------
// Bind a parsed ArgResult into psz_ctx.
// ---------------------------------------------------------------------------

static bool predictor_from_name(string const& v, psz_predictor& out)
{
  if (v == "spl-y25" or v == "spline-y25" or v == "spl" or v == "spline")
    out = psz_predictor::SplineY25;  // 2D+3D, ATT
  else if (v == "spl-y24" or v == "spline-y24")
    out = psz_predictor::SplineY24;  // 3D only
  else if (v == "lorenzo" or v == "lrz")
    out = psz_predictor::Lorenzo;
  else if (v == "lorenzo-zigzag" or v == "lrz-zz")
    out = psz_predictor::LorenzoZigZag;
  else
    return false;
  return true;
}

// "lc" names the pass-1 TCMS chain, and in a pass-2 slot it asks compose to pick the chain
static bool codec_from_name(string const& v, psz_codec& out)
{
  if (v == "hf" or v == "huffman" or v == "hf-rev2")
    out = psz_codec::HF_r2;  // HF_r2 supersedes HF
  else if (v == "hfr-v2" or v == "hfr-conservative")
    out = psz_codec::HFR;
  else if (v == "hfr-v3" or v == "hfr-direct") {
    cerr << LOG_WARN << "hfr-v3 is deprecated; use hfr-v4 (the default)." << endl;
    out = psz_codec::HFR_V3;
  }
  else if (v == "hfr-v4")
    out = psz_codec::HFR_V4;
  else if (v == "hfr-pbkc" or v == "hfr-pbk-compat" or v == "pbkc")
    out = psz_codec::HFR_PBKC;
  else if (v == "hfr-pbkgo" or v == "hfr-pbk-go" or v == "pbkgo")
    out = psz_codec::HFR_PBKGO;
  else if (v == "fzgcodec" or v == "fzg")
    out = psz_codec::FZG;
  else if (v == "drh" or v == "lc-drh")
    out = psz_codec::LC_DRH;
  else if (v == "tcms" or v == "lc-tcms" or v == "lc")
    out = psz_codec::LC_TCMS;
  else if (v == "bitr" or v == "lc-bitr")
    out = psz_codec::LC_BITR;
  else if (v == "rtr" or v == "lc-rtr")
    out = psz_codec::LC_RTR;
  else
    return false;
  return true;
}

static char const* predictor_name(psz_predictor p)
{
  switch (p) {
    case psz_predictor::Lorenzo: return "lrz";
    case psz_predictor::LorenzoZigZag: return "lrz-zz";
    case psz_predictor::SplineY24: return "spl-y24";
    case psz_predictor::SplineY25: return "spl-y25";
    default: return "?";
  }
}

static char const* codec_name(psz_codec c)
{
  switch (c) {
    case psz_codec::HF: return "hf";
    case psz_codec::HF_r2: return "hf-rev2";
    case psz_codec::HFR: return "hfr-v2";
    case psz_codec::HFR_V2: return "hfr-v2-raw";
    case psz_codec::HFR_V3: return "hfr-v3";
    case psz_codec::HFR_V4: return "hfr-v4";
    case psz_codec::HFR_PBKC: return "hfr-pbkc";
    case psz_codec::HFR_PBKGO: return "hfr-pbkgo";
    case psz_codec::HFR_PBKF: return "hfr-pbkf";
    case psz_codec::LC_TCMS: return "tcms";
    case psz_codec::LC_DRH: return "drh";
    case psz_codec::LC_BITR: return "bitr";
    case psz_codec::LC_RTR: return "rtr";
    case psz_codec::FZG: return "fzg";
    case psz_codec::CodecNull: return "none";
    default: return "?";
  }
}

static void psz_cli_bind(const _ptb::arg_result& args, psz_ctx* ctx)
{
  using namespace _ptb::detail;

  // input file
  apply_str(args.get<string>("input"), ctx->cli->file_input);

  // dimensions (xyz order)
  if (args.is_set("len")) {
    auto r             = parse_xyz(args.get<string>("len").c_str());
    ctx->header->len.x = r.len.x;
    ctx->header->len.y = r.len.y;
    ctx->header->len.z = r.len.z;
    ctx->len_linear    = r.len.x * r.len.y * r.len.z;
  }

  // dimensions (zyx order)
  if (args.is_set("len_zyx")) {
    auto r             = parse_zyx(args.get<string>("len_zyx").c_str());
    ctx->header->len.x = r.len.x;
    ctx->header->len.y = r.len.y;
    ctx->header->len.z = r.len.z;
    ctx->len_linear    = r.len.x * r.len.y * r.len.z;
  }

  // dtype
  {
    auto _v = args.get<string>("dtype");
    if (_v == "f32" or _v == "f4")
      ctx->header->dtype = F4;
    else if (_v == "f64" or _v == "f8")
      ctx->header->dtype = F8;
  }

  // error bound (kept as string to preserve exact repr in char_meta_eb)
  {
    auto _v = args.get<string>("eb");
    auto n  = str_to_num(_v.c_str());
    if (n) {
      ctx->header->eb = *n;
      apply_str(_v, ctx->cli->char_meta_eb);
    }
  }

  // mode
  {
    auto _v = args.get<string>("mode");
    if (not _v.empty()) {
      ctx->cli->rel_range_scan = (_v == "r2r" or _v == "rel");
      apply_str(_v, ctx->cli->char_mode);
    }
  }

  // predictor
  {
    auto _v = args.get<string>("pred");
    if (not _v.empty()) {
      apply_str(_v, ctx->cli->char_predictor_name);
      if (not predictor_from_name(_v, ctx->header->pipeline.predictor))
        printf("[psz::warning] \"%s\" unknown predictor; fallback to lorenzo.\n", _v.c_str());
    }
  }

  // histogram
  {
    auto _v = args.get<string>("hist");
    if (_v == "generic")
      ctx->header->pipeline.hist = psz_hist::HistGeneric;
    else if (_v == "sparse")
      ctx->header->pipeline.hist = psz_hist::HistSp;
  }

  // codec1: There is no standalone "hfr-v1".
  {
    auto _v = args.get<string>("codec1");
    if (not _v.empty()) {
      apply_str(_v, ctx->cli->char_codec1_name);
      if (not codec_from_name(_v, ctx->header->pipeline.codec1))
        printf("[psz::warning] \"%s\" unknown codec; keeping the default.\n", _v.c_str());
    }
  }

  // codec2
  {
    auto _v = args.get<string>("codec2");
    if (_v == "lc" or _v == "rtr" or _v == "bitr" or _v == "hi-cr" or _v == "hi-tp")
      ctx->header->pipeline.codec2 = psz_codec::LC_TCMS;
  }

  // Hi-mode config (--hi-config key=val,...)
  if (args.is_set("config"))
    hi_binder.bind(args.get<string>("config").c_str(), *CLI_interp_params(ctx));

  // report / dump flags
  if (args.is_set("report")) report_binder.bind(args.get<string>("report").c_str(), *ctx->cli);
  if (args.is_set("dump")) dump_binder.bind(args.get<string>("dump").c_str(), *ctx->cli);

  // skip
  {
    auto _v = args.get<string>("skip");
    if (_v.find("huffman") != string::npos) ctx->cli->skip_hf = true;
    if (_v.find("write2disk") != string::npos) ctx->cli->skip_tofile = true;
  }

  // compare / reference file
  apply_str(args.get<string>("compare"), ctx->cli->file_compare);

  // auto-tuning
  {
    auto _v = args.get<string>("auto");
    if (not _v.empty()) {
      if (_v == "cr-first" or _v == "CR-first")
        CLI_interp_params(ctx)->auto_tuning = 3;
      else if (_v == "rd-first" or _v == "RD-first")
        CLI_interp_params(ctx)->auto_tuning = 6;
      else {
        auto n = str_to_int(_v.c_str());
        if (not n) throw std::runtime_error("invalid auto-tuning value: " + _v);
        CLI_interp_params(ctx)->auto_tuning = static_cast<uint8_t>(*n);
      }
    }
  }

  // scheme shorthand
  {
    auto _v = args.get<string>("scheme");
    if (_v == "tp" or _v == "TP" or _v == "speed")
      ctx->header->pipeline.codec1 = LC_DRH;
    else if (_v == "cr" or _v == "CR")
      ctx->header->pipeline.codec1 = HFR_V4;
  }

  // pipeline: a whole pipeline by stage; overrides -p, -c1 and -c2
  {
    auto _v = args.get<string>("pipeline");
    if (not _v.empty()) {
      if (not args.get<string>("preset").empty()) {
        cerr << LOG_ERR << "--pipeline and --preset are mutually exclusive" << endl;
        exit(1);
      }

      std::vector<string> stage;
      for (size_t b = 0, e = 0; b <= _v.size(); b = e + 1) {
        e = _v.find_first_of(",;", b);
        if (e == string::npos) e = _v.size();
        stage.push_back(_v.substr(b, e - b));
      }

      if (stage.size() < 2 or stage.size() > 3) {
        cerr << LOG_ERR << "--pipeline takes 2 or 3 stages: p1,c1[,c2]" << endl;
        exit(1);
      }

      psz_predictor p1;
      psz_codec c1, c2 = psz_codec::CodecNull;
      if (not predictor_from_name(stage[0], p1)) {
        cerr << LOG_ERR << "no such predictor: " << stage[0] << endl;
        exit(1);
      }
      if (not codec_from_name(stage[1], c1)) {
        cerr << LOG_ERR << "no such codec: " << stage[1] << endl;
        exit(1);
      }
      if (stage.size() == 3 and not codec_from_name(stage[2], c2)) {
        cerr << LOG_ERR << "no such codec: " << stage[2] << endl;
        exit(1);
      }

      ctx->header->pipeline = pszppl_compose(p1, c1, c2);
    }
  }

  // preset: a whole pipeline by name; overrides -p, -c1 and -c2
  {
    auto _v = args.get<string>("preset");
    if (not _v.empty()) {
      psz_preset preset = PSZ_PRESET_HICR;
      bool known = true;
      if (_v == "fzg" or _v == "lrz-zz-fzg")
        preset = PSZ_PRESET_LRZZZ_FZG;
      else if (_v == "hicr" or _v == "hi-cr")
        preset = PSZ_PRESET_HICR;
      else if (_v == "hitp" or _v == "hi-tp")
        preset = PSZ_PRESET_HITP;
      else if (_v == "hitp_r1" or _v == "hitp-r1")
        preset = PSZ_PRESET_HITP_R1;
      else
        known = false;

      if (known) {
        ctx->header->pipeline = pszpreset_pipeline(preset);
        ctx->header->radius = pszpreset_radius(preset);
        ctx->bklen = ctx->header->radius * 2;
      }
      else
        printf("[psz::warning] \"%s\" unknown preset; ignored.\n", _v.c_str());
    }
  }

  // task flags (subcommand has priority over -z/-x)
  if (not ctx->cli->task_reduction and not ctx->cli->task_reconstruction) {
    if (args.get<bool>("compress")) ctx->cli->task_reduction = true;
    if (args.get<bool>("decompress")) ctx->cli->task_reconstruction = true;
  }

  if (args.get<bool>("verbose")) ctx->cli->verbose = true;
  if (args.get<bool>("hfd26")) ctx->cli->use_hfd26 = true;
  if (args.get<bool>("hfd_coarse")) ctx->cli->use_hfd_coarse = true;
  if (ctx->cli->use_hfd26 and ctx->cli->use_hfd_coarse) {
    cerr << LOG_ERR << "--hfd26 and --hfd-coarse select different decoders; pass at most one"
         << endl;
    exit(1);
  }

  // HFR reduce-merge pass count (--rmerge-count): 2|3|4; encode-only.
  // 0 means the flag was not given, and each codec keeps its own default.
  {
    auto const _rc = args.get<string>("rmerge_count");
    int const  v   = _rc.empty() ? 0 : (_rc == "2") ? 2 : (_rc == "3") ? 3 : (_rc == "4") ? 4 : -1;
    if (v < 0) {
      cerr << LOG_ERR << "--rmerge-count must be 2|3|4, got: " << _rc << endl;
      exit(1);
    }
    ctx->cli->hfr_rmerge_count = v;
  }

  // post-parse fixup: PBK variants and FZG bypass histogram
  // codec2==LC routes codec1 through plain Huffman_rev2
  if (ctx->header->pipeline.predictor == psz_predictor::LorenzoZigZag and
      (ctx->header->pipeline.codec1 == psz_codec::HFR_PBKC or
       ctx->header->pipeline.codec1 == psz_codec::HFR_PBKGO or
       ctx->header->pipeline.codec1 == psz_codec::HFR_V3 or
       ctx->header->pipeline.codec1 == psz_codec::HFR_V4 or
       ctx->header->pipeline.codec1 == psz_codec::HFR)) {
    cerr << LOG_ERR
         << "-p lrz-zz cannot pair with an HFR codec (hfr-v2, hfr-pbkc, hfr-pbkgo, hfr-v3, "
            "hfr-v4 [default]); use -p lrz, or --codec hf"
         << endl;
    exit(1);
  }

  if ((ctx->header->pipeline.codec1 == psz_codec::HFR_PBKC or
       ctx->header->pipeline.codec1 == psz_codec::HFR_PBKGO or
       ctx->header->pipeline.codec1 == psz_codec::FZG))
    ctx->header->pipeline.hist = psz_hist::HistNull;
}

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------

void pszctx_create_from_argv(psz_ctx* ctx, int const argc, char** const argv)
{
  // Detect optional subcommand at argv[1], build adjusted argv without it.
  int start = 1;
  if (argc > 1) {
    string first(argv[1]);
    if (first == "compress" or first == "comp" or first == "zip") {
      ctx->cli->task_reduction      = true;
      ctx->cli->task_reconstruction = false;
      start                         = 2;
    }
    else if (first == "decompress" or first == "decomp" or first == "unzip") {
      ctx->cli->task_reduction      = false;
      ctx->cli->task_reconstruction = true;
      start                         = 2;
    }
  }

  // Build adjusted argv (argv[0] + everything after the optional subcommand).
  std::vector<const char*> adj = {argv[0]};
  for (int k = start; k < argc; k++) adj.push_back(argv[k]);
  int adj_argc = static_cast<int>(adj.size());

  // Parse and bind. This function is declared `extern "C"`, so we MUST catch
  // any C++ exceptions here (throwing across the C boundary is UB). Exits with
  // psz_exit::USAGE (= 1) on parse / binder errors.
  if (adj_argc > 1) {
    try {
      auto args = psz_cli.parse(adj_argc, const_cast<char**>(adj.data()));
      psz_cli_bind(args, ctx);
    }
    catch (const std::runtime_error& e) {
      cerr << LOG_ERR << e.what() << endl;
      psz_print_document(false);
      exit(1);  // psz_exit::USAGE
    }
  }


  ctx->header->pipeline = pszppl_compose(
      ctx->header->pipeline.predictor, ctx->header->pipeline.codec1,
      ctx->header->pipeline.codec2);

  if (not pszppl_supported(ctx->header->pipeline)) {
    cerr << LOG_ERR << "unsupported pipeline: "
         << predictor_name(ctx->header->pipeline.predictor) << ","
         << codec_name(ctx->header->pipeline.codec1) << ","
         << codec_name(ctx->header->pipeline.codec2) << endl;
    exit(1);
  }

  if (ctx->header->pipeline.codec1 == psz_codec::LC_TCMS or
      ctx->header->pipeline.codec1 == psz_codec::LC_DRH)
    ctx->header->pipeline.hist = psz_hist::HistNull;

  if ((ctx->header->pipeline.codec1 == psz_codec::HFR_PBKC or
       ctx->header->pipeline.codec1 == psz_codec::HFR_PBKGO or
       ctx->header->pipeline.codec1 == psz_codec::HFR or
       ctx->header->pipeline.codec1 == psz_codec::HFR_V3 or
       ctx->header->pipeline.codec1 == psz_codec::HFR_V4)) {
    ctx->header->radius = 128;
    ctx->bklen             = 256;
  }
}

void psz_print_document(bool full)
{
  psz_version();
  std::cout << (full ? "\n" + _ptb::utils::doc_format(psz_full_doc)
                     : _ptb::utils::doc_format(psz_short_doc));
}

void pszctx_set_rawlen(psz_ctx* ctx, size_t _x, size_t _y, size_t _z)
{
  ctx->header->len.x = _x, ctx->header->len.y = _y, ctx->header->len.z = _z;

  ctx->len_linear = ctx->header->len.x * ctx->header->len.y * ctx->header->len.z;

  if (ctx->len_linear == 1)
    throw std::runtime_error("Input data length cannot be 1 (linearized).");
  if (ctx->len_linear == 0)
    throw std::runtime_error("Input data length cannot be 0 (linearized).");
}

psz_ctx* pszctx_default_values()
{
  return new psz_ctx{
      .header =
          new psz_header{
              .dtype = F4,
              {
                  .predictor = DEFAULT_PREDICTOR,
                  .hist      = DEFAULT_HISTOGRAM,
                  .codec1    = DEFAULT_CODEC,
                  .codec2    = CodecNull,
              },
              .eb     = 0.1,
              .radius = 512,
              .len =
                  {
                      .x = 1,
                      .y = 1,
                      .z = 1,
                  },
              .splen      = 0,
              .intp_param = make_default_params(),
          },
      .cli =
          new psz_cli_config{
              .dump_quantcode      = false,
              .dump_hist           = false,
              .task_reduction      = false,
              .task_reconstruction = false,
              .rel_range_scan      = false,
              .use_gpu_verify      = false,
              .skip_tofile         = false,
              .skip_hf             = false,
              .report_time         = false,
              .report_cr           = false,
              .verbose             = false,
              .use_hfd26           = false,
              .use_hfd_coarse      = false,
              .hfr_rmerge_count    = 0,
          },
      .bklen      = 1024,
      .len_linear = 1,
  };
}

psz_ctx* pszctx_minimal_workset(psz_dtype const dtype, psz_predictor const predictor,
                                int const quantizer_radius, psz_codec const codec)
{
  auto ws                        = pszctx_default_values();
  ws->header->dtype              = dtype;
  ws->header->pipeline.predictor = predictor;
  ws->header->pipeline.codec1    = codec;
  ws->bklen                      = quantizer_radius * 2;
  ws->header->radius          = quantizer_radius;
  return ws;
}

// clang-format off
unsigned int       CLI_x(psz_args* args)            { return args->header->len.x; }
unsigned int       CLI_y(psz_args* args)            { return args->header->len.y; }
unsigned int       CLI_z(psz_args* args)            { return args->header->len.z; }
unsigned short     CLI_radius(psz_args* args)       { return args->header->radius; }
unsigned short     CLI_bklen(psz_args* args)        { return args->header->radius * 2; }
psz_dtype          CLI_dtype(psz_args* args)        { return args->header->dtype; }
psz_predictor      CLI_predictor(psz_args* args)    { return args->header->pipeline.predictor; }
psz_ppl       CLI_pipeline(psz_args* args)     { return args->header->pipeline; }
psz_codec          CLI_codec1(psz_args* args)       { return args->header->pipeline.codec1; }
psz_codec          CLI_codec2(psz_args* args)       { return args->header->pipeline.codec2; }
psz_mode           CLI_mode(psz_args* args)         { return args->cli->rel_range_scan ? Rel : Abs; }
double             CLI_eb(psz_args* args)           { return args->header->eb; }
psz_interp_params* CLI_interp_params(psz_ctx* ctx)  { return &ctx->header->intp_param; }
// clang-format on
