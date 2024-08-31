/**
 * @file cmpare.thrust.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-08
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include "cusz/type.h"
#include "stat/compare.hh"

namespace psz::thrustgpu {

static const int MINVAL = 0;
static const int MAXVAL = 1;
static const int AVGVAL = 2;
static const int RNG = 3;

template <typename T>
void GPU_assess_quality(psz_summary* s, T* xdata, T* odata, size_t len)
{
  using tup = thrust::tuple<T, T>;

  thrust::device_ptr<T> p_odata =
      thrust::device_pointer_cast(odata);  // origin
  thrust::device_ptr<T> p_xdata = thrust::device_pointer_cast(xdata);

  T odata_res[4], xdata_res[4];

  // It takes too long to compile THRUST backend.
  // psz::probe_extrema<THRUST, T>(odata, len, odata_res);
  // psz::probe_extrema<THRUST, T>(xdata, len, xdata_res);
  psz::cuhip::GPU_extrema<T>(odata, len, odata_res);
  psz::cuhip::GPU_extrema<T>(xdata, len, xdata_res);

  auto begin = thrust::make_zip_iterator(thrust::make_tuple(p_odata, p_xdata));
  auto end = thrust::make_zip_iterator(
      thrust::make_tuple(p_odata + len, p_xdata + len));

  // clang-format off
    auto corr      = [=] __host__ __device__(tup t)  { return (thrust::get<0>(t) - odata_res[AVGVAL]) * (thrust::get<1>(t) - xdata_res[AVGVAL]); };
    auto err2      = []  __host__ __device__(tup t)  { T f = thrust::get<0>(t) - thrust::get<1>(t); return f * f; };
    auto var_odata = [=] __host__ __device__(T a) { T f = a - odata_res[AVGVAL]; return f * f; };
    auto var_xdata = [=] __host__ __device__(T a) { T f = a - xdata_res[AVGVAL]; return f * f; };

    auto sum_err2      = thrust::transform_reduce(begin, end, err2, 0.0f, thrust::plus<T>());
    auto sum_corr      = thrust::transform_reduce(begin, end, corr, 0.0f, thrust::plus<T>());
    auto sum_var_odata = thrust::transform_reduce(p_odata, p_odata + len, var_odata, 0.0f, thrust::plus<T>());
    auto sum_var_xdata = thrust::transform_reduce(p_xdata, p_xdata + len, var_xdata, 0.0f, thrust::plus<T>());
  // clang-format on

  double std_odata = sqrt(sum_var_odata / len);
  double std_xdata = sqrt(sum_var_xdata / len);
  double ee = sum_corr / len;

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
  s->score_MSE = sum_err2 / len;
  s->score_NRMSE = sqrt(s->score_MSE) / s->odata.rng;
  s->score_PSNR = 20 * log10(s->odata.rng) - 10 * log10(s->score_MSE);
}

}  // namespace psz::thrustgpu

#define __INSTANTIATE_THRUSTGPU_ASSESS(T)              \
  template void psz::thrustgpu::GPU_assess_quality<T>( \
      psz_summary * s, T * xdata, T * odata, size_t const len);
