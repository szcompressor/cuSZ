// SPDX-License-Identifier: BSD-3-Clause
// Predictor-run metrics: PSNR / NRMSE / max_err / outlier_count, plus
// `[key] value` emit-metrics formatter for the Phase 4 CLI.
//
// Pure host-side. Takes original + reconstructed arrays as raw pointers.

#ifndef PSZ_TEST_LIB_PRED_METRICS_HH
#define PSZ_TEST_LIB_PRED_METRICS_HH

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <string>

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
  bool has_nonfinite = false;
  size_t outlier_count = 0;
  size_t len = 0;
  std::string predictor;
  double eb = 0.0;
  int radius = 0;
};

inline PredMetrics compute_metrics(
    float const* orig, float const* reconstructed, size_t len,
    std::string predictor = {}, double eb = 0.0, int radius = 0,
    size_t outlier_count = 0)
{
  PredMetrics m;
  m.len = len;
  m.predictor = std::move(predictor);
  m.eb = eb;
  m.radius = radius;
  m.outlier_count = outlier_count;

  m.orig_min = std::numeric_limits<double>::infinity();
  m.orig_max = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < len; i++) {
    double const v = orig[i];
    if (v < m.orig_min) m.orig_min = v;
    if (v > m.orig_max) m.orig_max = v;
  }
  m.orig_range = m.orig_max - m.orig_min;

  double mse = 0;
  for (size_t i = 0; i < len; i++) {
    double const xv = reconstructed[i], ov = orig[i];
    if (!std::isfinite(xv) || !std::isfinite(ov)) {
      m.has_nonfinite = true;
      m.max_err_idx = i;
      break;
    }
    double const e = std::fabs(xv - ov);
    if (e > m.max_err) {
      m.max_err = e;
      m.max_err_idx = i;
    }
    mse += e * e;
  }
  m.mse = mse / (double)len;

  m.nrmse = (!m.has_nonfinite && m.orig_range > 0)
                ? (std::sqrt(m.mse) / m.orig_range)
                : std::numeric_limits<double>::quiet_NaN();
  m.psnr = (!m.has_nonfinite && m.mse > 0 && m.orig_range > 0)
               ? (20.0 * std::log10(m.orig_range) - 10.0 * std::log10(m.mse))
               : (m.has_nonfinite ? std::numeric_limits<double>::quiet_NaN()
                                  : std::numeric_limits<double>::infinity());
  return m;
}

// Human-friendly (default bin_pred output, 3 lines).
inline void print_human(const PredMetrics& m)
{
  printf(
      "[pred-study] predictor=%s  radius=%d  eb=%.4e  len=%zu\n",
      m.predictor.c_str(), m.radius, m.eb, m.len);
  printf(
      "[pred-study] quality  PSNR=%.8g  NRMSE=%.8g  max_err=%.8g  idx=%zu\n",
      m.psnr, m.nrmse, m.max_err, m.max_err_idx);
  printf(
      "[pred-study] outlier_count=%zu (%.4f%%)\n",
      m.outlier_count, m.len ? 100.0 * (double)m.outlier_count / (double)m.len : 0.0);
  if (m.has_nonfinite)
    printf("[pred-study] warning: non-finite value detected in reconstructed data\n");
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
  printf(
      "[outlier_pct]    %.6f\n",
      m.len ? 100.0 * (double)m.outlier_count / (double)m.len : 0.0);
  printf("[orig_range]    %.6e\n", m.orig_range);
}

// --assert-*: returns 0 if all asserts hold, 3 (assertion failure exit code)
// on first violation.
struct AssertConfig {
  double psnr_ge = -1.0;        // -1 = unset
  double max_err_le = -1.0;     // -1 = unset
  double max_err_rel_le = -1.0; // -1 = unset; max_err / orig_range
};

inline int check_asserts(const PredMetrics& m, const AssertConfig& a)
{
  if (a.psnr_ge >= 0 && m.psnr < a.psnr_ge) {
    fprintf(stderr,
            "[pred-study] assertion failed: psnr=%.6f < psnr_ge=%.6f\n",
            m.psnr, a.psnr_ge);
    return 3;
  }
  if (a.max_err_le >= 0 && m.max_err > a.max_err_le) {
    fprintf(stderr,
            "[pred-study] assertion failed: max_err=%.6e > max_err_le=%.6e\n",
            m.max_err, a.max_err_le);
    return 3;
  }
  if (a.max_err_rel_le >= 0) {
    double r = (m.orig_range > 0) ? (m.max_err / m.orig_range)
                                  : std::numeric_limits<double>::infinity();
    if (r > a.max_err_rel_le) {
      fprintf(stderr,
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
