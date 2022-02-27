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

#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <vector>

#include "../common.hh"
#include "format.hh"

using namespace std;

namespace analysis {

template <typename T>
void verify_data(stat_t* stat, T* xdata, T* odata, size_t len)
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

    stat->len = len;

    stat->max_odata = max_odata;
    stat->min_odata = min_odata;
    stat->rng_odata = max_odata - min_odata;
    stat->std_odata = std_odata;

    stat->max_xdata = max_xdata;
    stat->min_xdata = min_xdata;
    stat->rng_xdata = max_xdata - min_xdata;
    stat->std_xdata = std_xdata;

    stat->max_abserr_index  = max_abserr_index;
    stat->max_abserr        = max_abserr;
    stat->max_abserr_vs_rng = max_abserr / stat->rng_odata;
    stat->max_pwrrel_abserr = max_pwrrel_abserr;

    stat->coeff = ee / std_odata / std_xdata;
    stat->MSE   = sum_err2 / len;
    stat->NRMSE = sqrt(stat->MSE) / stat->rng_odata;
    stat->PSNR  = 20 * log10(stat->rng_odata) - 10 * log10(stat->MSE);
}

template <typename Data>
void print_data_quality_metrics(stat_t* stat, size_t compressed_bytes = 0, bool gpu_checker = false)
{
    auto checker = (not gpu_checker) ? string("(using CPU checker)") : string("(using GPU checker)");
    auto bytes   = (stat->len * sizeof(Data) * 1.0);

    auto print_ln3 = [](const char* s, double n1, double n2, double n3) {
        printf("  %-10s %16.8g %16.8g %16.8g %16s\n", s, n1, n2, n3, "");
    };

    auto print_ln = [](const char* s, double n1, double n2, double n3, double n4) {
        printf("  %-10s %16.8g %16.8g %16.8g %16.8g\n", s, n1, n2, n3, n4);
    };
    auto print_head = [](const char* s1, const char* s2, const char* s3, const char* s4, const char* s5) {
        printf("  \e[1m\e[31m%-10s %16s %16s %16s %16s\e[0m\n", s1, s2, s3, s4, s5);
    };

    printf("\nquality metrics %s:\n", checker.c_str());

    auto is_fp = std::is_same<Data, float>::value or std::is_same<Data, double>::value ? const_cast<char*>("yes")
                                                                                       : const_cast<char*>("no");
    print_head("", "data-len", "data-byte", "fp-type?", "");
    printf("  %-10s %16d %16d %16s\n", "", stat->len, sizeof(Data), is_fp);

    print_head("", "min", "max", "rng", "std");
    print_ln("origin", stat->min_odata, stat->max_odata, stat->rng_odata, stat->std_odata);
    print_ln("eb-lossy", stat->min_xdata, stat->max_xdata, stat->rng_xdata, stat->std_xdata);

    print_head("", "abs-val", "abs-idx", "pw-rel", "VS-RNG");
    print_ln("max-error", stat->max_abserr, stat->max_abserr_index, stat->max_pwrrel_abserr, stat->max_abserr_vs_rng);

    print_head("", "CR", "NRMSE", "corr-coeff", "PSNR");
    print_ln("metrics", bytes / compressed_bytes, stat->NRMSE, stat->coeff, stat->PSNR);

    printf("\n");
};

}  // namespace analysis

#endif