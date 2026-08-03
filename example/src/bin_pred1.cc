#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>

#include "compare.hh"
#include "compressor.hh"
#include "cusz.h"
#include "kernel.hh"
#include "ptb.hh"
#include "test_lib/pred_args.hh"

namespace utils = _ptb::utils;

int main(int argc, char** argv)
{
  psz_test::PredArgs args;
  int parse_rc = args.parse(argc, argv);
  if (args.help) {
    psz_test::PredArgs::usage(argv[0]);
    return 0;
  }
  if (parse_rc == 77) return 77;
  if (parse_rc != 0) {
    psz_test::PredArgs::usage(argv[0]);
    return 2;
  }

  // --cross-check targets bin_pred_xv (the spl-vN cross-validation driver),
  // not this single-predictor path.
  if (args.do_cross_check) {
    fprintf(
        stderr,
        "[pred-study] --cross-check is now bin_pred_xv (a separate driver).\n"
        "             Run:  bin_pred_xv %s\n",
        args.predictor.c_str());
    return 2;
  }

  psz_predictor pred_type;
  int spline_v = 0;
  if (not psz_test::resolve_predictor(args.predictor, pred_type, spline_v)) {
    fprintf(stderr, "[pred-study] unknown predictor: %s\n", args.predictor.c_str());
    return 2;
  }
  int spline_variant = (spline_v == 24) ? 1 : 0;  // 0 = y25 (2D+3D), 1 = y24 (lean 3D)

  std::string const& fname = args.fname;
  std::string const& pred_name = args.predictor;
  size_t x = args.x, y = args.y, z = args.z;
  size_t len = x * y * z;
  int radius = args.radius;
  bool do_export = args.do_export;

  auto h_data = MAKE_UNIQUE_HOST(float, len);
  auto d_data = MAKE_UNIQUE_DEVICE(float, len);
  utils::fromfile_or_die(fname, h_data.get(), len);
  memcpy_allkinds<H2D>(d_data.get(), h_data.get(), len);

  // resolve eb: --rel converts the user value against the data range.
  double const user_eb = args.eb;
  double abs_eb = user_eb;
  if (args.mode == psz_test::PredArgs::Mode::Rel) {
    double mn = h_data[0], mx = h_data[0];
    for (size_t i = 1; i < len; ++i) {
      double const v = h_data[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    abs_eb = user_eb * (mx - mn);
  }

  auto d_xdata = MAKE_UNIQUE_DEVICE(float, len);
  memset_device(d_xdata.get(), len);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto manager = psz_create_resource_manager(
      F4, {x, y, z}, {pred_type, HistGeneric, HF, CodecNull}, spline_variant, (void*)stream);

  manager->header->rc.eb = abs_eb;
  manager->header->rc.mode = (args.mode == psz_test::PredArgs::Mode::Rel) ? Rel : Abs;
  manager->header->rc.radius = radius;
  manager->header->user_input_eb = user_eb;

  using E = uint16_t;
  using M = uint32_t;
  using PPL = psz::compression_pipeline<float, E>;
  using Buf = psz_buf<float, E>;

  auto mem = (Buf*)manager->buf;
  auto h_hist = MAKE_UNIQUE_HOST(uint32_t, manager->bklen);

  auto status = PPL::compress_analysis(manager, mem, d_data.get(), h_hist.get(), (void*)stream);
  if (status != PSZ_SUCCESS) {
    printf("[pred-study] predictor-analysis failed, status=%d\n", status);
    psz_release_resource(manager);
    cudaStreamDestroy(stream);
    return 2;
  }
  cudaStreamSynchronize(stream);

  // use CodecNull to skip codec: bridge (f4)eq into the fused source the x-side reads.
  auto const _l = manager->header->len;
  bool const tile_nd = (_l.y > 1);  // 2D/3D predictors tile; 1D is linear == tile
  size_t const n_fused = tile_nd ? mem->eq_len() : len;
  {
    auto h_eq = MAKE_UNIQUE_HOST(uint16_t, n_fused);
    auto h_space = MAKE_UNIQUE_HOST(float, n_fused);
    memcpy_allkinds<D2H>(h_eq.get(), mem->eq_d(), n_fused);
    for (size_t i = 0; i < n_fused; ++i) h_space[i] = (float)h_eq[i];
    if (tile_nd) {
      mem->alloc_decode_fused();
      memcpy_allkinds<H2D>(mem->decode_fused_d(), h_space.get(), n_fused);
    }
    else
      memcpy_allkinds<H2D>(d_xdata.get(), h_space.get(), n_fused);
  }

  // reverse predictor. overflow outliers carry tile gids -> scatter into the
  // fused source pre-x (mirrors the codec decode); linear keeps the pre-x scatter into out.
  if (tile_nd)
    psz::module::GPU_scatter<float, M>::kernel_v3_fuse(
        (_ptb::compact_cell<float, M>*)mem->outlier2_validx_d(), (int)manager->header->splen,
        mem->decode_fused_d(), (void*)stream);
  else
    PPL::decomp_scatter(
        manager->header, (_ptb::compact_cell<float, M>*)mem->outlier2_validx_d(), d_xdata.get(),
        (void*)stream);
  PPL::decomp_predict(manager->header, mem, mem->anchor_d(), d_xdata.get(), (void*)stream);

  cudaStreamSynchronize(stream);

  size_t const outlier_count = manager->header->splen;
  double const outlier_pct = len ? 100.0 * (double)outlier_count / (double)len : 0.0;

  psz_stats stat{};
  psz::analysis::assess_quality<CUDA, float>(&stat, d_xdata.get(), d_data.get(), len);

  printf(
      "[pred-study] predictor=%s  radius=%d  eb=%.4e  len=%zu\n", pred_name.c_str(), radius,
      abs_eb, len);
  printf(
      "[pred-study] quality  PSNR=%.8g  NRMSE=%.8g  max_err=%.8g  idx=%zu\n", stat.score_PSNR,
      stat.score_NRMSE, stat.max_err_abs, stat.max_err_idx);
  printf("[pred-study] outlier_count=%zu (%.4f%%)\n", outlier_count, outlier_pct);

  if (do_export) {
    auto h_eq = MAKE_UNIQUE_HOST(uint16_t, len);
    memcpy_allkinds<D2H>(h_eq.get(), mem->eq_d(), len);
    std::string eq_out = fname + ".pred_" + pred_name + ".ectrl.u2";
    utils::tofile(eq_out, h_eq.get(), len);
    printf("[pred-study] ectrl written to: %s\n", eq_out.c_str());

    auto h_xdata = MAKE_UNIQUE_HOST(float, len);
    memcpy_allkinds<D2H>(h_xdata.get(), d_xdata.get(), len);
    std::string rec_out = fname + ".pred_" + pred_name + ".rec.f4";
    utils::tofile(rec_out, h_xdata.get(), len);
    printf("[pred-study] reconstructed written to: %s\n", rec_out.c_str());

    if (pred_type == psz_predictor::Spline) {
      auto anchor_len = mem->anchor_len();
      auto h_anchor = MAKE_UNIQUE_HOST(float, anchor_len);
      memcpy_allkinds<D2H>(h_anchor.get(), mem->anchor_d(), anchor_len);
      std::string anc_out = fname + ".pred_spline.anchor.f4";
      utils::tofile(anc_out, h_anchor.get(), anchor_len);
      printf("[pred-study] anchor(%zu) written to: %s\n", anchor_len, anc_out.c_str());
    }
  }

  // Machine-readable [key] value block for ctest / scrapers (bin_hf contract).
  if (args.emit_metrics) {
    printf("\n");
    printf("[predictor]      %s\n", pred_name.c_str());
    printf("[eb]             %.6e\n", abs_eb);
    printf("[radius]         %d\n", radius);
    printf("[len]            %zu\n", len);
    printf("[psnr]           %.6f\n", stat.score_PSNR);
    printf("[nrmse]          %.6e\n", stat.score_NRMSE);
    printf("[max_err]        %.6e\n", stat.max_err_abs);
    printf("[max_err_idx]    %zu\n", stat.max_err_idx);
    printf("[outlier_count]  %zu\n", outlier_count);
    printf("[outlier_pct]    %.6f\n", outlier_pct);
    printf("[orig_range]    %.6e\n", stat.odata.rng);
  }

  // --assert-*: exit 3 on the first violation (-1 thresholds are unset).
  int assert_rc = 0;
  {
    auto const& a = args.asserts;
    if (a.psnr_ge >= 0 and stat.score_PSNR < a.psnr_ge) {
      fprintf(
          stderr, "[pred-study] assertion failed: psnr=%.6f < psnr_ge=%.6f\n", stat.score_PSNR,
          a.psnr_ge);
      assert_rc = 3;
    }
    else if (a.max_err_le >= 0 and stat.max_err_abs > a.max_err_le) {
      fprintf(
          stderr, "[pred-study] assertion failed: max_err=%.6e > max_err_le=%.6e\n",
          stat.max_err_abs, a.max_err_le);
      assert_rc = 3;
    }
    else if (a.max_err_rel_le >= 0) {
      double const r = (stat.odata.rng > 0) ? (stat.max_err_abs / stat.odata.rng)
                                            : std::numeric_limits<double>::infinity();
      if (r > a.max_err_rel_le) {
        fprintf(
            stderr,
            "[pred-study] assertion failed: max_err/range=%.6e > max_err_rel_le=%.6e "
            "(max_err=%.6e, range=%.6e)\n",
            r, a.max_err_rel_le, stat.max_err_abs, stat.odata.rng);
        assert_rc = 3;
      }
    }
  }

  psz_release_resource(manager);
  cudaStreamDestroy(stream);
  return assert_rc;
}
