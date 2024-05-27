/**
 * @file compare.stl.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-08
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef C0E747B4_066F_4B04_A3D2_00E1A3B7D682
#define C0E747B4_066F_4B04_A3D2_00E1A3B7D682

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>

#include "cusz/type.h"

namespace psz {

template <typename T>
bool cppstl_identical(T* d1, T* d2, size_t const len)
{
  return std::equal(d1, d1 + len, d2);
}

template <typename T>
void cppstl_extrema(T* in, szt const len, T res[4])
{
  auto res2 = std::minmax_element(in, in + len);
  res[0] = *res2.first;
  res[1] = *res2.second;
  res[3] = res[1] - res[0];  // range

  auto sum = std::accumulate(in, in + len, (T)0.0);
  res[2] = sum / len;  // average
}

template <typename T>
bool cppstl_error_bounded(
    T* a, T* b, size_t const len, double const eb,
    size_t* first_faulty_idx = nullptr)
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
void cppstl_assess_quality(psz_summary* s, T* xdata, T* odata, size_t const len)
{
  double max_odata = odata[0], min_odata = odata[0];
  double max_xdata = xdata[0], min_xdata = xdata[0];
  double max_abserr = max_abserr = fabs(xdata[0] - odata[0]);

  double sum_0 = 0, sum_x = 0;
  for (size_t i = 0; i < len; i++) sum_0 += odata[i], sum_x += xdata[i];

  double mean_odata = sum_0 / len, mean_xdata = sum_x / len;
  double sum_var_odata = 0, sum_var_xdata = 0, sum_err2 = 0, sum_corr = 0,
         rel_abserr = 0;

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
      max_pwrrel_abserr =
          max_pwrrel_abserr < rel_abserr ? rel_abserr : max_pwrrel_abserr;
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

  s->max_err.idx = max_abserr_index;
  s->max_err.abs = max_abserr;
  s->max_err.rel = max_abserr / s->odata.rng;
  s->max_err.pwrrel = max_pwrrel_abserr;

  s->score.coeff = ee / std_odata / std_xdata;
  s->score.MSE = sum_err2 / len;
  s->score.NRMSE = sqrt(s->score.MSE) / s->odata.rng;
  s->score.PSNR = 20 * log10(s->odata.rng) - 10 * log10(s->score.MSE);
}

}  // namespace psz

#endif /* C0E747B4_066F_4B04_A3D2_00E1A3B7D682 */
