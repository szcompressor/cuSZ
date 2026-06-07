// CLI argument parser shared by bin_pred and bin_pred_xv. Built on the
// portable arg_builder so we get consistent help output, type-safe access,
// and unified parsing across the project's binaries.
//
// Two invocation modes:
//   1. Registry mode (preferred for ctest):
//        bin_pred --config <toml> --dataset <key> [--predictor <name>] [flags...]
//      Dimensions, eb, and file path come from the TOML registry stanza.
//      The ctest matrix uses this exclusively.
//
//   2. Direct mode (ad-hoc):
//        bin_pred --input <file> --xyz <NxMxK> --eb <eb> [--predictor <name>] [flags...]
//      All sources explicit; no registry lookup.

#ifndef PSZ_TEST_LIB_PRED_ARGS_HH
#define PSZ_TEST_LIB_PRED_ARGS_HH

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "arg_builder.hh"   // _ptb::arg_builder
#include "pred_metrics.hh"  // AssertConfig
#include "toml_lite.hh"

namespace psz_test {

struct PredArgs {
  enum class Mode { Abs, Rel };

  // input + dims + bound
  std::string fname;
  size_t x = 0, y = 0, z = 1;
  double eb = 0.0;
  Mode mode = Mode::Abs;
  std::string predictor = "spline";
  int radius = 128;

  // registry / source flags
  std::string config_path;
  std::string dataset_key;
  bool require_file = false;

  // not-yet-implemented (parsed only)
  std::string synth_spec;

  // output / behavior flags
  bool emit_metrics = false;
  bool do_export = false;
  bool do_cross_check = false;

  // assertions (from --assert-* flags)
  AssertConfig asserts;

  bool help = false;

  static void usage(const char* /*prog*/) { /* arg_builder prints its own */ }

  int parse(int argc, char** argv)
  {
    // 1. split kv-pair
    std::vector<std::string> argv_owned;
    std::vector<char*> argv_view;
    argv_owned.reserve(argc * 2);
    for (int i = 0; i < argc; ++i) {
      std::string t(argv[i]);
      if (i > 0 && t.size() > 2 && t[0] == '-' && t[1] == '-') {
        auto eq = t.find('=');
        if (eq != std::string::npos) {
          argv_owned.push_back(t.substr(0, eq));
          argv_owned.push_back(t.substr(eq + 1));
          continue;
        }
      }
      argv_owned.emplace_back(std::move(t));
    }
    for (auto& s : argv_owned) argv_view.push_back(const_cast<char*>(s.c_str()));
    int new_argc = static_cast<int>(argv_view.size());
    char** new_argv = argv_view.data();

    // Detect early -h/--help so we don't error out on missing required args.
    for (int i = 1; i < new_argc; ++i) {
      std::string a(new_argv[i]);
      if (a == "-h" || a == "--help") {
        help = true;
        return 0;
      }
    }

    // 2. parse args
    // clang-format off
    auto cli =
        _ptb::arg_builder("bin_pred / bin_pred_xv")
            .string ("input",       {"-i", "--input"},               "",       "data file (direct mode)")
            .string ("xyz",         {"-l", "--xyz", "--len"},        "",       "3-D dims NxMxK (direct mode)")
            .number ("eb",          {"-e", "--eb"},                  0.0,      "error bound")
            .string ("predictor",   {"-p", "--predictor"},           "spline", "predictor name")
            .integer("radius",      {"-r", "--radius"},              128,      "lookup radius")
            .string ("mode",        {"--mode"},                      "abs",    "abs|rel")
            .flag   ("rel_flag",    {"--rel"},                                 "shorthand for --mode rel")
            .flag   ("abs_flag",    {"--abs"},                                 "shorthand for --mode abs")
            .string ("config",      {"--config"},                    "",       "TOML registry path")
            .string ("dataset",     {"--dataset"},                   "",       "registry stanza key")
            .flag   ("require_file",{"--require-file"},                        "exit 77 if data file absent")
            .string ("synth",       {"--synth"},                     "",       "(reserved; errors out)")
            .flag   ("emit_metrics",{"--emit-metrics"},                        "print [key] value block")
            .flag   ("cross_check", {"--cross-check", "--xcheck"},             "bin_pred -> bin_pred_xv hint")
            .flag   ("do_export",   {"--export"},                              "dump ectrl (and anchor) files")
            .number ("assert_psnr_ge",        {"--assert-psnr-ge"},       -1.0, "psnr floor")
            .number ("assert_max_err_le",     {"--assert-max-err-le"},    -1.0, "max_err ceiling")
            .number ("assert_max_err_rel_le", {"--assert-max-err-rel-le"},-1.0, "max_err/range ceiling");
    // clang-format on

    // 3. parse
    _ptb::arg_result r;
    try {
      r = cli.parse(new_argc, new_argv);
    }
    catch (std::exception& e) {
      fprintf(stderr, "[pred-study] %s\n", e.what());
      return 2;
    }

    // 4. unpack into pred args
    fname = r.get<std::string>("input");
    eb = r.get<f8>("eb");
    predictor = r.get<std::string>("predictor");
    radius = static_cast<int>(r.get<i8>("radius"));
    config_path = r.get<std::string>("config");
    dataset_key = r.get<std::string>("dataset");
    require_file = r.get<bool>("require_file");
    synth_spec = r.get<std::string>("synth");
    emit_metrics = r.get<bool>("emit_metrics");
    do_cross_check = r.get<bool>("cross_check");
    do_export = r.get<bool>("do_export");

    // mode: --rel/--abs
    bool rel_flag = r.get<bool>("rel_flag");
    bool abs_flag = r.get<bool>("abs_flag");
    if (rel_flag and abs_flag) {
      fprintf(stderr, "[pred-study] --rel and --abs are mutually exclusive\n");
      return 2;
    }
    if (rel_flag)
      mode = Mode::Rel;
    else if (abs_flag)
      mode = Mode::Abs;
    else {
      auto m = r.get<std::string>("mode");
      if (m == "abs" or m == "Abs")
        mode = Mode::Abs;
      else if (m == "rel" or m == "Rel" or m == "r2r")
        mode = Mode::Rel;
      else {
        fprintf(stderr, "[pred-study] --mode: expected abs|rel, got %s\n", m.c_str());
        return 2;
      }
    }

    // dimensions: parse --xyz string if non-empty.
    {
      auto xyz_str = r.get<std::string>("xyz");
      if (not xyz_str.empty()) {
        auto pl = _ptb::detail::parse_xyz(xyz_str.c_str()).len;
        x = pl.x, y = pl.y, z = pl.z;
      }
    }

    // assertions: -1.0 sentinel = unset.
    asserts.psnr_ge = r.get<f8>("assert_psnr_ge");
    asserts.max_err_le = r.get<f8>("assert_max_err_le");
    asserts.max_err_rel_le = r.get<f8>("assert_max_err_rel_le");

    // 5. resolve registry
    if (not dataset_key.empty()) {
      std::string p = resolve_registry_path(config_path);
      if (p.empty()) {
        fprintf(
            stderr,
            "[pred-study] --dataset given but no registry found "
            "(--config / $PSZ_TEST_DATA / ~/.psz_test_data.toml)\n");
        return 2;
      }
      Registry reg;
      try {
        reg = parse_file(p);
      }
      catch (std::exception& e) {
        fprintf(stderr, "[pred-study] registry parse error: %s\n", e.what());
        return 2;
      }
      auto d = lookup_dataset(reg, dataset_key);
      if (not d) {
        fprintf(
            stderr, "[pred-study] dataset '%s' not in registry %s\n", dataset_key.c_str(),
            p.c_str());
        return 2;
      }
      if (fname.empty()) fname = d->path;
      if (x == 0) x = (size_t)d->x;
      if (y == 0) y = (size_t)d->y;
      if (z == 0 or z == 1) z = (size_t)d->z;
      if (eb == 0.0 and not r.is_set("eb")) eb = d->eb;
    }

    // 6. validation
    if (fname.empty() or x == 0 or y == 0 or z == 0 or eb == 0.0) {
      fprintf(
          stderr,
          "[pred-study] missing required input (use either --dataset <key>\n"
          "             or --input <file> --xyz <NxMxK> --eb <eb>)\n");
      return 2;
    }

    // --synth TODO: parsed but errors out.
    if (not synth_spec.empty()) {
      fprintf(
          stderr,
          "[pred-study] --synth is reserved for future use; smooth-field\n"
          "             generators for lossy predictors aren't implemented.\n"
          "             See doc/2026-05-09_synth-lossy-todo.md.\n");
      return 2;
    }

    // --require-file: existence check.
    if (require_file and not file_exists(fname)) {
      fprintf(
          stderr, "[pred-study] --require-file: %s missing -> exit 77 (skip)\n", fname.c_str());
      return 77;
    }

    return 0;
  }

  size_t total_len() const { return x * y * z; }
};

}  // namespace psz_test

#endif
