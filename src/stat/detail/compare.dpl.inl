/**
 * @file _compare.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-08
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef F7DF2FE5_571E_48C1_965D_0B19D1CC14D4
#define F7DF2FE5_571E_48C1_965D_0B19D1CC14D4

#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

#include "stat/compare/compare.dpl.hh"

// #include <thrust/count.h>
// #include <thrust/iterator/constant_iterator.h>
#include <cmath>

#include "cusz/type.h"
#include "stat/compare.hh"

namespace psz {

static const int MINVAL = 0;
static const int MAXVAL = 1;
static const int AVGVAL = 2;
static const int RNG = 3;

template <typename T>
void dpl_assess_quality(psz_summary* s, T* xdata, T* odata, size_t len)
{
  using tup = std::tuple<T, T>;

  dpct::device_pointer<T> p_odata = dpct::get_device_pointer(odata);  // origin
  dpct::device_pointer<T> p_xdata = dpct::get_device_pointer(xdata);

  T odata_res[4], xdata_res[4];

  // thrustgpu_get_extrema_rawptr(odata, len, odata_res);
  // thrustgpu_get_extrema_rawptr(xdata, len, xdata_res);
  psz::probe_extrema<ONEAPI, T>(odata, len, odata_res);
  psz::probe_extrema<ONEAPI, T>(xdata, len, xdata_res);

  auto begin =
      oneapi::dpl::make_zip_iterator(std::make_tuple(p_odata, p_xdata));
  auto end = oneapi::dpl::make_zip_iterator(
      std::make_tuple(p_odata + len, p_xdata + len));

  // clang-format off
  auto corr      = [=] (tup t)  { return (std::get<0>(t) - odata[AVGVAL]) * (std::get<1>(t) - xdata[AVGVAL]); };
  auto err2      = []  (tup t)  { T f = std::get<0>(t) - std::get<1>(t); return f * f; };
  auto var_odata = [=] (T a) { T f = a - odata[AVGVAL]; return f * f; };
  auto var_xdata = [=] (T a) { T f = a - xdata[AVGVAL]; return f * f; };

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

  s->odata.max = odata_res[MAXVAL];
  s->odata.min = odata_res[MINVAL];
  s->odata.rng = odata_res[MAXVAL] - odata_res[MINVAL];
  s->odata.std = std_odata;

  s->xdata.max = xdata_res[MAXVAL];
  s->xdata.min = xdata_res[MINVAL];
  s->xdata.rng = xdata_res[MAXVAL] - xdata_res[MINVAL];
  s->xdata.std = std_xdata;

  s->max_err.idx = max_abserr_index;
  s->max_err.abs = max_abserr;
  s->max_err.rel = max_abserr / s->odata.rng;
  s->max_err.pwrrel = NAN;

  s->score.coeff = ee / std_odata / std_xdata;
  s->score.MSE = sum_err2 / len;
  s->score.NRMSE = sqrt(s->score.MSE) / s->odata.rng;
  s->score.PSNR = 20 * log10(s->odata.rng) - 10 * log10(s->score.MSE);
}

}  // namespace psz

#endif /* F7DF2FE5_571E_48C1_965D_0B19D1CC14D4 */
