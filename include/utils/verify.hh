#ifndef ANALYSIS_VERIFY_HH
#define ANALYSIS_VERIFY_HH

/**
 * @file verify.hh
 * @author Jiannan Tian
 * @brief Verification of decompressed data.
 * @version 0.2
 * @date 2020-09-20
 * Created on: 2019-09-30
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include "../cusz/type.h"
#include "../utils/config.hh"

using namespace std;

namespace cusz {

template <typename T>
void verify_data(psz_summary* s, T* xdata, T* odata, size_t len)
{
    double max_odata = odata[0], min_odata = odata[0];
    double max_xdata = xdata[0], min_xdata = xdata[0];
    double max_abserr = max_abserr = fabs(xdata[0] - odata[0]);

    double sum_0 = 0, sum_x = 0;
    for (size_t i = 0; i < len; i++) sum_0 += odata[i], sum_x += xdata[i];

    double mean_odata = sum_0 / len, mean_xdata = sum_x / len;
    double sum_var_odata = 0, sum_var_xdata = 0, sum_err2 = 0, sum_corr = 0, rel_abserr = 0;

    double max_pwrrel_abserr = 0;
    size_t max_abserr_index  = 0;
    for (size_t i = 0; i < len; i++) {
        max_odata = max_odata < odata[i] ? odata[i] : max_odata;
        min_odata = min_odata > odata[i] ? odata[i] : min_odata;

        max_xdata = max_xdata < odata[i] ? odata[i] : max_xdata;
        min_xdata = min_xdata > xdata[i] ? xdata[i] : min_xdata;

        float abserr = fabs(xdata[i] - odata[i]);
        if (odata[i] != 0) {
            rel_abserr        = abserr / fabs(odata[i]);
            max_pwrrel_abserr = max_pwrrel_abserr < rel_abserr ? rel_abserr : max_pwrrel_abserr;
        }
        max_abserr_index = max_abserr < abserr ? i : max_abserr_index;
        max_abserr       = max_abserr < abserr ? abserr : max_abserr;
        sum_corr += (odata[i] - mean_odata) * (xdata[i] - mean_xdata);
        sum_var_odata += (odata[i] - mean_odata) * (odata[i] - mean_odata);
        sum_var_xdata += (xdata[i] - mean_xdata) * (xdata[i] - mean_xdata);
        sum_err2 += abserr * abserr;
    }
    double std_odata = sqrt(sum_var_odata / len);
    double std_xdata = sqrt(sum_var_xdata / len);
    double ee        = sum_corr / len;

    s->len = len;

    s->odata.max = max_odata;
    s->odata.min = min_odata;
    s->odata.rng = max_odata - min_odata;
    s->odata.std = std_odata;

    s->xdata.max = max_xdata;
    s->xdata.min = min_xdata;
    s->xdata.rng = max_xdata - min_xdata;
    s->xdata.std = std_xdata;

    s->max_err.idx    = max_abserr_index;
    s->max_err.abs    = max_abserr;
    s->max_err.rel    = max_abserr / s->odata.rng;
    s->max_err.pwrrel = max_pwrrel_abserr;

    s->score.coeff = ee / std_odata / std_xdata;
    s->score.MSE   = sum_err2 / len;
    s->score.NRMSE = sqrt(s->score.MSE) / s->odata.rng;
    s->score.PSNR  = 20 * log10(s->odata.rng) - 10 * log10(s->score.MSE);
}

}  // namespace cusz

#endif
