/**
 * @file verify.cc
 * @author Jiannan Tian
 * @brief Verification of decompressed data.
 * @version 0.1
 * @date 2020-09-20
 * Created on: 2019-09-30
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cstdio>
#include <limits>
#include <numeric>
#include <vector>

#include "format.hh"
#include "types.hh"
#include "verify.hh"

using namespace std;

template <typename T>
void analysis::VerifyData(stat_t* stat, T* xData, T* oData, size_t _len)
{
    double _max = 0, _min = 0, max_abserr = 0;
    _max = oData[0], _min = oData[0];
    max_abserr = xData[0] > oData[0] ? xData[0] - oData[0] : oData[0] - xData[0];

    double sum_0 = 0, sum_x = 0;
    for (size_t i = 0; i < _len; i++) sum_0 += oData[i], sum_x += xData[i];

    double mean_0 = sum_0 / _len, mean_x = sum_x / _len;
    double sum_var_0 = 0, sum_var_x = 0, sum_sq_err = 0, sum_corre = 0, rel_abserr = 0;

    double max_pwr_rel_abserr = 0;
    size_t max_abserr_index   = 0;
    for (size_t i = 0; i < _len; i++) {
        _max = _max < oData[i] ? oData[i] : _max, _min = _min > oData[i] ? oData[i] : _min;
        float abserr = fabs(xData[i] - oData[i]);
        if (oData[i] != 0) {
            rel_abserr         = abserr / fabs(oData[i]);
            max_pwr_rel_abserr = max_pwr_rel_abserr < rel_abserr ? rel_abserr : max_pwr_rel_abserr;
        }
        max_abserr_index = max_abserr < abserr ? i : max_abserr_index;
        max_abserr       = max_abserr < abserr ? abserr : max_abserr;
        sum_corre += (oData[i] - mean_0) * (xData[i] - mean_x);
        sum_var_0 += (oData[i] - mean_0) * (oData[i] - mean_0);
        sum_var_x += (xData[i] - mean_x) * (xData[i] - mean_x);
        sum_sq_err += abserr * abserr;
    }
    double std_0 = sqrt(sum_var_0 / _len);
    double std_x = sqrt(sum_var_x / _len);
    double ee    = sum_corre / _len;

    stat->len                 = _len;
    stat->coeff               = ee / std_0 / std_x;
    stat->maximum             = _max;
    stat->minimum             = _min;
    stat->range               = _max - _min;
    stat->max_abserr_index    = max_abserr_index;
    stat->max_abserr          = max_abserr;
    stat->max_abserr_vs_range = max_abserr / stat->range;
    stat->max_pwr_rel_abserr  = max_pwr_rel_abserr;
    stat->MSE                 = sum_sq_err / _len;
    stat->NRMSE               = sqrt(stat->MSE) / stat->range;
    stat->PSNR                = 20 * log10(stat->range) - 10 * log10(stat->MSE);
}

template void analysis::VerifyData<float>(stat_t*, float*, float*, size_t);

void analysis::PrintMetrics(
    stat_t* stat,
    int     type_byte,
    bool    override_eb,
    double  new_eb,
    size_t  archive_byte,
    size_t  bin_scale)
{
    auto indent  = []() { printf("  "); };
    auto newline = []() { printf("\n"); };
    cout << "\n";
    indent(), printf("%-20s%.20G", "min.val", stat->minimum), newline();
    indent(), printf("%-20s%.20G", "max.val", stat->maximum), newline();
    indent(), printf("%-20s%.20G", "val.rng", stat->range), newline();
    indent(), printf("%-20s\e[31m%.20G\e[0m", "max.err.abs.val", stat->max_abserr), newline();
    indent(), printf("%-20s%lu", "max.err.abs.idx", stat->max_abserr_index), newline();
    if (override_eb) {
        indent(), printf("----------------------------------------------------------------\n");
        indent(), printf("OVERRODE eb (because of, for example, binning) to:\t%.6G", new_eb), newline();
        indent(), printf("max.err.abs.val/OVERRIDEN eb:\t%.6G", stat->max_abserr / new_eb), newline();
        indent(), printf("----------------------------------------------------------------\n");
    }
    else {
        indent(), printf("%-20s\e[31m%.20G\e[0m", "max.err.vs.rng", stat->max_abserr_vs_range), newline();
    }
    indent(), printf("%-20s%.20G", "max.pw.rel.err", stat->max_pwr_rel_abserr), newline();
    indent(), printf("%-20s\e[31m%.20G\e[0m", "PSNR", stat->PSNR), newline();
    indent(), printf("%-20s%.20G", "NRMSE", stat->NRMSE), newline();
    indent(), printf("%-20s%.20G", "correl.coeff", stat->coeff), newline();
    if (archive_byte) {
        indent(),
            printf(
                "%-20s\e[31m%lf\e[0m", "comp.ratio.w/o.gzip", bin_scale * 1.0 * stat->len * type_byte / archive_byte),
            newline();
    }
    cout << endl;
};
