#include <cmath>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

#include "compare.hh"
#include "cusz/type.h"

namespace psz::dpl {

static const int MINVAL = 0;
static const int MAXVAL = 1;
static const int AVGVAL = 2;
static const int RNG = 3;

template <typename T>
void GPU_assess_quality(psz_statistics* s, T* xdata, T* odata, size_t len)
{
  static_assert(std::is_same_v<T, f4>, "No f8 for local GPU; fast fail on sycl::aspects::fp64.");

  using tup = std::tuple<T, T>;

  dpct::device_pointer<T> p_odata = dpct::get_device_pointer(odata);  // origin
  dpct::device_pointer<T> p_xdata = dpct::get_device_pointer(xdata);

  auto [odata_min, odata_max, odata_avg, odata_rng] = psz::GPU_probe_extrema<T, SYCL>(odata, len);
  auto [xdata_min, xdata_max, xdata_avg, xdata_rng] = psz::GPU_probe_extrema<T, SYCL>(xdata, len);

  auto begin = oneapi::dpl::make_zip_iterator(std::make_tuple(p_odata, p_xdata));
  auto end = oneapi::dpl::make_zip_iterator(std::make_tuple(p_odata + len, p_xdata + len));

  // clang-format off
  auto corr      = [=] (tup t)  { return (std::get<0>(t) - odata_avg) * (std::get<1>(t) - xdata_avg); };
  auto err2      = []  (tup t)  { T f = std::get<0>(t) - std::get<1>(t); return f * f; };
  auto var_odata = [=] (T a) { T f = a - odata_avg; return f * f; };
  auto var_xdata = [=] (T a) { T f = a - xdata_avg; return f * f; };

  auto sum_err2      = std::transform_reduce(oneapi::dpl::execution::seq, begin, end, 0.0f, std::plus<T>(), err2);
  auto sum_corr      = std::transform_reduce(oneapi::dpl::execution::seq, begin, end, 0.0f, std::plus<T>(), corr);
  auto sum_var_odata = std::transform_reduce(oneapi::dpl::execution::seq, p_odata, p_odata + len, 0.0f, std::plus<T>(), var_odata);
  auto sum_var_xdata = std::transform_reduce(oneapi::dpl::execution::seq, p_xdata, p_xdata + len, 0.0f, std::plus<T>(), var_xdata);
  // clang-format on

  double std_odata = sqrt(sum_var_odata / len);
  double std_xdata = sqrt(sum_var_xdata / len);
  double ee = sum_corr / len;

  // -----------------------------------------------------------------------------
  T max_abserr{0};
  size_t max_abserr_index{0};
  psz::dpl_get_maxerr(xdata, odata, len, max_abserr, max_abserr_index, false);
  // -----------------------------------------------------------------------------

  s->len = len;

  s->odata.max = odata_max;
  s->odata.min = odata_min;
  s->odata.rng = odata_rng;
  s->odata.avg = odata_avg;
  s->odata.std = std_odata;

  s->xdata.max = xdata_max;
  s->xdata.min = xdata_min;
  s->xdata.rng = xdata_rng;
  s->xdata.avg = xdata_avg;
  s->xdata.std = std_xdata;

  s->max_err_idx = max_abserr_index;
  s->max_err_abs = max_abserr;
  s->max_err_rel = max_abserr / s->odata.rng;
  s->max_err_pwrrel = NAN;

  s->score_coeff = ee / std_odata / std_xdata;
  s->score_MSE = sum_err2 / len;
  s->score_NRMSE = sqrt(s->score_MSE) / s->odata.rng;
  s->score_PSNR = 20 * log10(s->odata.rng) - 10 * log10(s->score_MSE);
}

}  // namespace psz::dpl

#define __INSTANTIATE_DPL_ASSESS(T)              \
  template void psz::dpl::GPU_assess_quality<T>( \
      psz_statistics * s, T * xdata, T * odata, size_t const len);
