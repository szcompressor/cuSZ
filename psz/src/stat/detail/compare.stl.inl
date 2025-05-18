/**
 * @file compare.stl.inl
 * @author Jiannan Tian
 * @brief Verification of decompressed data.
 * @version 0.2
 * @date 2020-09-20
 * Created on: 2019-09-30
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory See LICENSE in top-level directory
 *
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>

#include "cusz/type.h"
#include "detail/compare.hh"

namespace psz::cppstl {

bool CPU_identical(void* d1, void* d2, size_t sizeof_T, size_t const len)
{
  return std::equal((u1*)d1, (u1*)d1 + len * sizeof_T, (u1*)d2);
}

template <typename T>
void CPU_extrema(T* in, szt const len, T res[4])
{
  auto res2 = std::minmax_element(in, in + len);
  res[0] = *res2.first;
  res[1] = *res2.second;
  res[3] = res[1] - res[0];  // range

  auto sum = std::accumulate(in, in + len, (T)0.0);
  res[2] = sum / len;  // average
}

template <typename T>
bool CPU_error_bounded(T* a, T* b, size_t const len, double const eb, size_t* first_faulty_idx)
{
  // debugging

  bool eb_ed = true;
  for (size_t i = 0; i < len; i++) {
    if (fabs(a[i] - b[i]) > 1.001 * eb) {
      if (first_faulty_idx) *first_faulty_idx = i;
      return false;
    }
  }
  return true;
}

template <typename T>
void CPU_find_max_error(T* a, T* b, size_t const len, T& maxval, size_t& maxloc)
{
  maxval = T(0);
  maxloc = 0;

  // Find the maximum error and its index
  for (size_t i = 0; i < len; ++i) {
    T error = std::fabs(a[i] - b[i]);
    if (error > maxval) maxval = error, maxloc = i;
  }
}

template <typename T>
void CPU_assess_quality(psz_statistics* s, T* xdata, T* odata, size_t const len)
{
  double max_odata = odata[0], min_odata = odata[0];
  double max_xdata = xdata[0], min_xdata = xdata[0];
  double max_abserr = max_abserr = fabs(xdata[0] - odata[0]);

  double sum_0 = 0, sum_x = 0;
  for (size_t i = 0; i < len; i++) sum_0 += odata[i], sum_x += xdata[i];

  double mean_odata = sum_0 / len, mean_xdata = sum_x / len;
  double sum_var_odata = 0, sum_var_xdata = 0, sum_err2 = 0, sum_corr = 0, rel_abserr = 0;

  double max_pwrrel_abserr = 0;
  size_t max_abserr_index = 0;
  for (size_t i = 0; i < len; i++) {
    max_odata = max_odata < odata[i] ? odata[i] : max_odata;
    min_odata = min_odata > odata[i] ? odata[i] : min_odata;

    max_xdata = max_xdata < odata[i] ? odata[i] : max_xdata;
    min_xdata = min_xdata > xdata[i] ? xdata[i] : min_xdata;

    float abserr = fabs(xdata[i] - odata[i]);
    if (odata[i] != 0) {
      rel_abserr = abserr / fabs(odata[i]);
      max_pwrrel_abserr = max_pwrrel_abserr < rel_abserr ? rel_abserr : max_pwrrel_abserr;
    }
    max_abserr_index = max_abserr < abserr ? i : max_abserr_index;
    max_abserr = max_abserr < abserr ? abserr : max_abserr;
    sum_corr += (odata[i] - mean_odata) * (xdata[i] - mean_xdata);
    sum_var_odata += (odata[i] - mean_odata) * (odata[i] - mean_odata);
    sum_var_xdata += (xdata[i] - mean_xdata) * (xdata[i] - mean_xdata);
    sum_err2 += abserr * abserr;
  }
  double std_odata = sqrt(sum_var_odata / len);
  double std_xdata = sqrt(sum_var_xdata / len);
  double ee = sum_corr / len;

  s->len = len;

  s->odata.max = max_odata;
  s->odata.min = min_odata;
  s->odata.rng = max_odata - min_odata;
  s->odata.std = std_odata;

  s->xdata.max = max_xdata;
  s->xdata.min = min_xdata;
  s->xdata.rng = max_xdata - min_xdata;
  s->xdata.std = std_xdata;

  s->max_err_idx = max_abserr_index;
  s->max_err_abs = max_abserr;
  s->max_err_rel = max_abserr / s->odata.rng;
  s->max_err_pwrrel = max_pwrrel_abserr;

  s->score_coeff = ee / std_odata / std_xdata;
  s->score_MSE = sum_err2 / len;
  s->score_NRMSE = sqrt(s->score_MSE) / s->odata.rng;
  s->score_PSNR = 20 * log10(s->odata.rng) - 10 * log10(s->score_MSE);
}

}  // namespace psz::cppstl
