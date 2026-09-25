#include <cuda_runtime.h>

#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "cusz.h"
#include "test_lib/pred_args.hh"

template <typename V>
static bool fromfile(std::string const& fname, V* p, size_t n)
{
  std::ifstream in(fname, std::ios::binary);
  return bool(in.read(reinterpret_cast<char*>(p), n * sizeof(V)));
}

template <typename V>
static void tofile(std::string const& fname, V const* p, size_t n)
{ std::ofstream(fname, std::ios::binary).write(reinterpret_cast<char const*>(p), n * sizeof(V)); }

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
  if (not psz_test::resolve_predictor(args.predictor, pred_type)) {
    fprintf(stderr, "[pred-study] unknown predictor: %s\n", args.predictor.c_str());
    return 2;
  }

  std::string const& fname = args.fname;
  std::string const& pred_name = args.predictor;
  size_t x = args.x, y = args.y, z = args.z;
  size_t len = x * y * z;
  bool do_export = args.do_export;

  std::vector<float> h_data(len);
  if (not fromfile(fname, h_data.data(), len)) {
    fprintf(stderr, "[pred-study] failed to read \"%s\"\n", fname.c_str());
    return 2;
  }
  float *d_data, *d_xdata;
  cudaMalloc(&d_data, len * sizeof(float));
  cudaMalloc(&d_xdata, len * sizeof(float));
  cudaMemcpy(d_data, h_data.data(), len * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_xdata, 0, len * sizeof(float));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto compressor = psz_compress_init(F4, {x, y, z}, (void*)stream);

  // resolve eb: --rel converts the user value against the data range.
  double const user_eb = args.eb;
  double abs_eb = user_eb;
  if (args.mode == psz_test::PredArgs::Mode::Rel)
    abs_eb *= psz_compress_extrema_float(compressor, d_data).rng;

  psz_header header;
  uint8_t* d_archive = nullptr;
  size_t archive_bytes = 0;
  auto status = psz_compress_process_float(
      compressor, {pred_type, HistGeneric, CodecNull, CodecNull}, abs_eb, d_data);
  if (status == PSZ_SUCCESS)
    status = psz_compress_archive(compressor, &header, &d_archive, &archive_bytes);
  if (status != PSZ_SUCCESS) {
    printf("[pred-study] predictor-analysis failed, status=%d\n", status);
    psz_free(compressor);
    cudaStreamDestroy(stream);
    return 2;
  }

  auto decompressor = psz_decompress_init(&header, (void*)stream);
  status = psz_decompress_process_float(decompressor, d_archive, archive_bytes, d_xdata);
  if (status != PSZ_SUCCESS) {
    printf("[pred-study] predictor-reconstruction failed, status=%d\n", status);
    psz_free(decompressor);
    psz_free(compressor);
    cudaStreamDestroy(stream);
    return 2;
  }
  cudaStreamSynchronize(stream);

  size_t const outlier_count = header.splen;
  double const outlier_pct = len ? 100.0 * (double)outlier_count / (double)len : 0.0;

  psz_stats stat{};
  psz_assess_quality_float(&stat, d_xdata, d_data, len);

  printf(
      "[pred-study] predictor=%s  radius=%d  eb=%.4e  len=%zu\n", pred_name.c_str(), header.radius,
      abs_eb, len);
  printf(
      "[pred-study] quality  PSNR=%.8g  NRMSE=%.8g  max_err=%.8g  idx=%zu\n", stat.score.PSNR,
      stat.score.NRMSE, stat.max_err.abs, stat.max_err.idx);
  printf("[pred-study] outlier_count=%zu (%.4f%%)\n", outlier_count, outlier_pct);

  if (do_export) {
    std::vector<uint8_t> h_archive(archive_bytes);
    cudaMemcpy(h_archive.data(), d_archive, archive_bytes, cudaMemcpyDeviceToHost);

    std::string eq_out = fname + ".pred_" + pred_name + ".ectrl.u2";
    tofile(eq_out, (uint16_t const*)(h_archive.data() + header.entry[PSZHEADER_ENCODED]), len);
    printf("[pred-study] ectrl written to: %s\n", eq_out.c_str());

    std::vector<float> h_xdata(len);
    cudaMemcpy(h_xdata.data(), d_xdata, len * sizeof(float), cudaMemcpyDeviceToHost);
    std::string rec_out = fname + ".pred_" + pred_name + ".rec.f4";
    tofile(rec_out, h_xdata.data(), len);
    printf("[pred-study] reconstructed written to: %s\n", rec_out.c_str());

    size_t const anchor_len =
        (header.entry[PSZHEADER_SPFMT] - header.entry[PSZHEADER_ANCHOR]) / sizeof(float);
    if (anchor_len != 0) {
      std::string anc_out = fname + ".pred_spline.anchor.f4";
      tofile(
          anc_out, (float const*)(h_archive.data() + header.entry[PSZHEADER_ANCHOR]), anchor_len);
      printf("[pred-study] anchor(%zu) written to: %s\n", anchor_len, anc_out.c_str());
    }
  }

  // Machine-readable [key] value block for ctest / scrapers (bin_hf contract).
  if (args.emit_metrics) {
    printf("\n");
    printf("[predictor]      %s\n", pred_name.c_str());
    printf("[eb]             %.6e\n", abs_eb);
    printf("[radius]         %d\n", header.radius);
    printf("[len]            %zu\n", len);
    printf("[psnr]           %.6f\n", stat.score.PSNR);
    printf("[nrmse]          %.6e\n", stat.score.NRMSE);
    printf("[max_err]        %.6e\n", stat.max_err.abs);
    printf("[max_err_idx]    %zu\n", stat.max_err.idx);
    printf("[outlier_count]  %zu\n", outlier_count);
    printf("[outlier_pct]    %.6f\n", outlier_pct);
    printf("[orig_range]    %.6e\n", stat.odata.rng);
  }

  // --assert-*: exit 3 on the first violation (-1 thresholds are unset).
  int assert_rc = 0;
  {
    auto const& a = args.asserts;
    if (a.psnr_ge >= 0 and stat.score.PSNR < a.psnr_ge) {
      fprintf(
          stderr, "[pred-study] assertion failed: psnr=%.6f < psnr_ge=%.6f\n", stat.score.PSNR,
          a.psnr_ge);
      assert_rc = 3;
    }
    else if (a.max_err_le >= 0 and stat.max_err.abs > a.max_err_le) {
      fprintf(
          stderr, "[pred-study] assertion failed: max_err=%.6e > max_err_le=%.6e\n",
          stat.max_err.abs, a.max_err_le);
      assert_rc = 3;
    }
    else if (a.max_err_rel_le >= 0) {
      double const r = (stat.odata.rng > 0) ? (stat.max_err.abs / stat.odata.rng)
                                            : std::numeric_limits<double>::infinity();
      if (r > a.max_err_rel_le) {
        fprintf(
            stderr,
            "[pred-study] assertion failed: max_err/range=%.6e > max_err_rel_le=%.6e "
            "(max_err=%.6e, range=%.6e)\n",
            r, a.max_err_rel_le, stat.max_err.abs, stat.odata.rng);
        assert_rc = 3;
      }
    }
  }

  psz_free(decompressor);
  psz_free(compressor);
  cudaFree(d_data);
  cudaFree(d_xdata);
  cudaStreamDestroy(stream);
  return assert_rc;
}
