/**
 * @file bin_pred.cc
 *
 * Single-predictor microbench: forward + reverse predictor (analysis-only,
 * no codec), reports PSNR / NRMSE / max_err / outlier_count.
 *
 * The v1<->vN cross-validation lives in `bin_pred_xv` (Phase 3 of the test
 * infra migration; see doc/2026-05-09_test-infra-migration-bin_pred.md).
 *
 * Building blocks live in `test_lib/`:
 *   PredArgs   (pred_args.hh)    — CLI parser, registry resolution
 *   PredRun    (pred_run.hh)     — load + manager init + compress + reconstruct
 *   PredMetrics (pred_metrics.hh) — metric compute + emit-metrics + asserts
 */

#include <cstdio>
#include <memory>
#include <string>

#include "test_lib/pred_args.hh"
#include "test_lib/pred_metrics.hh"
#include "test_lib/pred_run.hh"

namespace _utils = _portable::utils;

using psz_test::PredArgs;
using psz_test::PredMetrics;
using psz_test::PredRun;

// EXPORT: dump ectrl (and anchor for Spline) to <fname>.pred_<predictor>.*
static void run_export(PredRun& run)
{
  using E = PredRun::E;
  size_t const len = run.len;
  auto h_ectrl = std::unique_ptr<E[]>(new E[len]);
  cudaMemcpy(h_ectrl.get(), run.mem->ectrl_d(), sizeof(E) * len, cudaMemcpyDeviceToHost);
  std::string ectrl_out = run.args.fname + ".pred_" + run.args.predictor + ".ectrl.u2";
  _utils::tofile(ectrl_out, h_ectrl.get(), len);
  printf("[pred-study] ectrl written to: %s\n", ectrl_out.c_str());

  if (run.pred_type == psz_predictor::Spline) {
    auto anchor_len = run.mem->anchor_len();
    auto h_anchor = std::unique_ptr<float[]>(new float[anchor_len]);
    cudaMemcpy(h_anchor.get(), run.mem->anchor_d(), sizeof(float) * anchor_len, cudaMemcpyDeviceToHost);
    std::string anc_out = run.args.fname + ".pred_spline.anchor.f4";
    _utils::tofile(anc_out, h_anchor.get(), anchor_len);
    printf("[pred-study] anchor(%zu) written to: %s\n", anchor_len, anc_out.c_str());
  }
}

int main(int argc, char** argv)
{
  PredArgs args;
  int parse_rc = args.parse(argc, argv);
  if (args.help) { PredArgs::usage(argv[0]); return 0; }
  if (parse_rc == 77) return 77;
  if (parse_rc != 0) { PredArgs::usage(argv[0]); return 2; }

  // bin_pred is the single-predictor microbench. If the user asked for a
  // spl-vN target with --cross-check, point them at bin_pred_xv (the Phase-3
  // dedicated driver that handles the v1↔vN comparison).
  if (args.do_cross_check) {
    fprintf(stderr,
            "[pred-study] --cross-check is now bin_pred_xv (a separate driver).\n"
            "             Run:  bin_pred_xv %s\n",
            args.predictor.c_str());
    return 2;
  }

  PredRun run(args);
  if (int e = run.setup(); e != 0) return e;
  if (int e = run.compress(); e != 0) return e;
  run.reconstruct_v1_path();
  PredMetrics m = run.compute_metrics();

  psz_test::print_human(m);

  if (args.do_export) run_export(run);

  if (args.emit_metrics) {
    printf("\n");
    psz_test::print_emit_metrics(m);
  }

  if (int rc = psz_test::check_asserts(m, args.asserts); rc != 0) return rc;
  return 0;
}
