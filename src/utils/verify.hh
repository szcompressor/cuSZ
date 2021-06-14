#ifndef UTILS_VERIFY_HH
#define UTILS_VERIFY_HH

/**
 * @file verify.cc
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

#include <cstdio>
#include <limits>
#include <numeric>
#include <vector>

#include "../types.hh"
#include "format.hh"

using namespace std;

namespace analysis {

template <typename T>
void VerifyData(stat_t* stat, T* xdata, T* odata, size_t len)
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

    stat->len               = len;
    stat->max_odata         = max_odata;
    stat->min_odata         = min_odata;
    stat->rng_odata         = max_odata - min_odata;
    stat->std_odata         = std_odata;
    stat->max_xdata         = max_xdata;
    stat->min_xdata         = min_xdata;
    stat->rng_xdata         = max_xdata - min_xdata;
    stat->std_xdata         = std_xdata;
    stat->coeff             = ee / std_odata / std_xdata;
    stat->max_abserr_index  = max_abserr_index;
    stat->max_abserr        = max_abserr;
    stat->max_abserr_vs_rng = max_abserr / stat->rng_odata;
    stat->max_pwrrel_abserr = max_pwrrel_abserr;
    stat->MSE               = sum_err2 / len;
    stat->NRMSE             = sqrt(stat->MSE) / stat->rng_odata;
    stat->PSNR              = 20 * log10(stat->rng_odata) - 10 * log10(stat->MSE);
}

template <typename Data>
void PrintMetrics(
    stat_t* stat,
    bool    override_eb  = false,
    double  new_eb       = 0,
    size_t  archive_byte = 0,
    size_t  bin_scale    = 1,
    bool    locate_err   = false,
    bool    gpu_checker  = false)
{
    auto indent   = []() {};  //{ printf("  "); };
    auto linkline = []() { printf("|\n"); };
    auto newline  = []() { printf("\n"); };

    if (not gpu_checker)
        cout << ">> CPU checker\n\n";
    else
        cout << ">> GPU checker\n\n";

    indent(), printf("%-20s%15lu", "data.len", stat->len), newline();
    linkline();
    indent(), printf("%-20s%15.8g", "min.odata", stat->min_odata), newline();
    indent(), printf("%-20s%15.8g", "max.odata", stat->max_odata), newline();
    indent(), printf("%-20s%15.8g", "rng.odata", stat->rng_odata), newline();
    indent(), printf("%-20s%15.8g", "std.odata", stat->std_odata), newline();
    linkline();
    indent(), printf("%-20s%15.8g", "min.xdata", stat->min_xdata), newline();
    indent(), printf("%-20s%15.8g", "max.xdata", stat->max_xdata), newline();
    indent(), printf("%-20s%15.8g", "rng.xdata", stat->rng_xdata), newline();
    indent(), printf("%-20s%15.8g", "std.xdata", stat->std_xdata), newline();

    if (locate_err) {
        linkline();
        indent(), printf("%-20s\e[31m%15.8g\e[0m", "max.err.abs.val", stat->max_abserr), newline();
        indent(), printf("%-20s%15lu", "max.err.abs.idx", stat->max_abserr_index), newline();
        if (override_eb) {
            indent(), printf("----------------------------------------------------------------\n");
            indent(), printf("OVERRODE eb (because of, for example, binning) to:\t%.6G", new_eb), newline();
            indent(), printf("max.err.abs.val/OVERRIDEN eb:\t%.6G", stat->max_abserr / new_eb), newline();
            indent(), printf("----------------------------------------------------------------\n");
        }
        else {
            indent(), printf("%-20s\e[31m%15.8e\e[0m", "max.err.vs.rng", stat->max_abserr_vs_rng), newline();
        }
        indent(), printf("%-20s%15.8g", "max.pw.rel.err", stat->max_pwrrel_abserr), newline();
    }

    linkline();
    indent(), printf("%-20s\e[31m%15.4f\e[0m", "PSNR", stat->PSNR), newline();
    indent(), printf("%-20s%15.8g", "NRMSE", stat->NRMSE), newline();
    indent(), printf("%-20s%15.8g", "corr.coeff", stat->coeff), newline();

    if (archive_byte) {
        indent(),
            printf("%-20s\e[31m%15.8lf\e[0m", "CR.no.gzip", bin_scale * 1.0 * stat->len * sizeof(Data) / archive_byte),
            newline();
    }
    cout << endl;
};

}  // namespace analysis

#endif