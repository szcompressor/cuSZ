#include "stat/compare.hh"

namespace psz::cuhip {

constexpr auto MINVAL = 0;
constexpr auto MAXVAL = 1;
constexpr auto AVGVAL = 2;
constexpr auto RNG = 3;

constexpr auto SUM_CORR = 0;
constexpr auto SUM_ERR_SQ = 1;
constexpr auto SUM_VAR_ODATA = 2;
constexpr auto SUM_VAR_XDATA = 3;

template <typename T>
void GPU_assess_quality(psz_summary* s, T* xdata, T* odata, size_t const len)
{
  T odata_res[4], xdata_res[4];

  psz::cuhip::GPU_extrema<T>(odata, len, odata_res);
  psz::cuhip::GPU_extrema<T>(xdata, len, xdata_res);

  T h_err[4];

  psz::cuhip::GPU_calculate_errors<T>(
      odata, odata_res[AVGVAL], xdata, xdata_res[AVGVAL], len, h_err);

  double std_odata = sqrt(h_err[SUM_VAR_ODATA] / len);
  double std_xdata = sqrt(h_err[SUM_VAR_XDATA] / len);
  double ee = h_err[SUM_CORR] / len;

  // -----------------------------------------------------------------------------
  T max_abserr{0};
  size_t max_abserr_index{0};
  psz::thrustgpu::GPU_max_error(
      xdata, odata, len, max_abserr, max_abserr_index, false);
  // -----------------------------------------------------------------------------

  s->len = len;

  s->odata.max = odata_res[MAXVAL];
  s->odata.min = odata_res[MINVAL];
  s->odata.rng = odata_res[MAXVAL] - odata_res[MINVAL];
  s->odata.avg = odata_res[AVGVAL];
  s->odata.std = std_odata;

  s->xdata.max = xdata_res[MAXVAL];
  s->xdata.min = xdata_res[MINVAL];
  s->xdata.rng = xdata_res[MAXVAL] - xdata_res[MINVAL];
  s->xdata.avg = xdata_res[AVGVAL];
  s->xdata.std = std_xdata;

  s->max_err_idx = max_abserr_index;
  s->max_err_abs = max_abserr;
  s->max_err_rel = max_abserr / s->odata.rng;
  s->max_err_pwrrel = NAN;

  s->score_coeff = ee / std_odata / std_xdata;
  s->score_MSE = h_err[SUM_ERR_SQ] / len;
  s->score_NRMSE = sqrt(s->score_MSE) / s->odata.rng;
  s->score_PSNR = 20 * log10(s->odata.rng) - 10 * log10(s->score_MSE);
}

}  // namespace psz::cu_hip

#define __INSTANTIATE_CUHIP_ASSESS(T)              \
  template void psz::cuhip::GPU_assess_quality<T>( \
      psz_summary * s, T * xdata, T * odata, size_t const len);
