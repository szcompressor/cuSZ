#ifndef PSZ_TEST_LIB_PRED_METRICS_HH
#define PSZ_TEST_LIB_PRED_METRICS_HH

#include <cstddef>
#include <cstdio>
#include <limits>
#include <string>

#include "compare.hh"

namespace psz_test {

struct PredMetrics {
  double psnr = 0.0;
  double nrmse = 0.0;
  double max_err = 0.0;
  size_t max_err_idx = 0;
  double mse = 0.0;
  double orig_min = 0.0;
  double orig_max = 0.0;
  double orig_range = 0.0;
  size_t outlier_count = 0;
  size_t len = 0;
  std::string predictor;
  double eb = 0.0;
  int radius = 0;
};

inline PredMetrics compute_metrics(
    float const* orig, float const* reconstructed, size_t len, std::string predictor = {},
    double eb = 0.0, int radius = 0, size_t outlier_count = 0)
{
  PredMetrics m;
  m.len = len;
  m.predictor = std::move(predictor);
  m.eb = eb;
  m.radius = radius;
  m.outlier_count = outlier_count;

  psz_stats s{};
  psz::analysis::assess_quality<SEQ, float>(
      &s, const_cast<float*>(reconstructed), const_cast<float*>(orig), len);

  m.psnr = s.score_PSNR;
  m.nrmse = s.score_NRMSE;
  m.max_err = s.max_err_abs;
  m.max_err_idx = s.max_err_idx;
  m.mse = s.score_MSE;
  m.orig_min = s.odata.min;
  m.orig_max = s.odata.max;
  m.orig_range = s.odata.rng;
  return m;
}

// Human-friendly (default bin_pred output, 3 lines).
inline void print_human(const PredMetrics& m)
{
  printf(
      "[pred-study] predictor=%s  radius=%d  eb=%.4e  len=%zu\n", m.predictor.c_str(), m.radius,
      m.eb, m.len);
  printf(
      "[pred-study] quality  PSNR=%.8g  NRMSE=%.8g  max_err=%.8g  idx=%zu\n", m.psnr, m.nrmse,
      m.max_err, m.max_err_idx);
  printf(
      "[pred-study] outlier_count=%zu (%.4f%%)\n", m.outlier_count,
      m.len ? 100.0 * (double)m.outlier_count / (double)m.len : 0.0);
}

// Machine-readable `[key] value` block for ctest / scrapers.
// Mirrors the bin_hf [key] value contract.
inline void print_emit_metrics(const PredMetrics& m)
{
  printf("[predictor]      %s\n", m.predictor.c_str());
  printf("[eb]             %.6e\n", m.eb);
  printf("[radius]         %d\n", m.radius);
  printf("[len]            %zu\n", m.len);
  printf("[psnr]           %.6f\n", m.psnr);
  printf("[nrmse]          %.6e\n", m.nrmse);
  printf("[max_err]        %.6e\n", m.max_err);
  printf("[max_err_idx]    %zu\n", m.max_err_idx);
  printf("[outlier_count]  %zu\n", m.outlier_count);
  printf("[outlier_pct]    %.6f\n", m.len ? 100.0 * (double)m.outlier_count / (double)m.len : 0.0);
  printf("[orig_range]    %.6e\n", m.orig_range);
}

// --assert-*: returns 0 if all asserts hold, 3 (assertion failure exit code)
// on first violation.
struct AssertConfig {
  double psnr_ge = -1.0;         // -1 = unset
  double max_err_le = -1.0;      // -1 = unset
  double max_err_rel_le = -1.0;  // -1 = unset; max_err / orig_range
};

inline int check_asserts(const PredMetrics& m, const AssertConfig& a)
{
  if (a.psnr_ge >= 0 && m.psnr < a.psnr_ge) {
    fprintf(
        stderr, "[pred-study] assertion failed: psnr=%.6f < psnr_ge=%.6f\n", m.psnr, a.psnr_ge);
    return 3;
  }
  if (a.max_err_le >= 0 && m.max_err > a.max_err_le) {
    fprintf(
        stderr, "[pred-study] assertion failed: max_err=%.6e > max_err_le=%.6e\n", m.max_err,
        a.max_err_le);
    return 3;
  }
  if (a.max_err_rel_le >= 0) {
    double r =
        (m.orig_range > 0) ? (m.max_err / m.orig_range) : std::numeric_limits<double>::infinity();
    if (r > a.max_err_rel_le) {
      fprintf(
          stderr,
          "[pred-study] assertion failed: max_err/range=%.6e > max_err_rel_le=%.6e "
          "(max_err=%.6e, range=%.6e)\n",
          r, a.max_err_rel_le, m.max_err, m.orig_range);
      return 3;
    }
  }
  return 0;
}

}  // namespace psz_test

#endif
