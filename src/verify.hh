#ifndef VERIFY_HH
#define VERIFY_HH

#include <cstdio>
#include <limits>
#include <numeric>
#include <vector>

#include "format.hh"
#include "types.hh"

using namespace std;

// template <typename T>
// void verify_data2(T* xData_POD, T* oData_POD, size_t __len, size_t c_byte_size = 0) {
//    std::vector<double> xData(xData_POD, xData_POD + __len);
//    std::vector<double> oData(oData_POD, oData_POD + __len);
//
//    double max_val   = *std::max_element(oData.begin(), oData.end());
//    double min_val   = *std::min_element(oData.begin(), oData.end());
//    double val_range = max_val - min_val;
//
//    std::vector<double> err;
//    std::transform(oData.begin(), oData.end(), xData.begin(), std::back_inserter(err), std::minus<T>());
//
//    std::vector<double> abserr(err);  // TODO change element from n to e
//    std::for_each(abserr.begin(), abserr.end(), [](double& e) { e = fabs(e); });
//    double max_abserr = *std::max_element(abserr.begin(), abserr.end());
//
//    double sum_0 = std::accumulate(oData.begin(), oData.end(), 0.0);
//    double sum_x = std::accumulate(xData.begin(), xData.end(), 0.0);
//
//    double mean_0 = sum_0 / __len;
//    double mean_x = sum_x / __len;
//
//    vector<double> var_0(oData);
//    vector<double> var_x(xData);
//    vector<double> sq_err(err);
//
//    std::for_each(var_0.begin(), var_0.end(), [&](double& e) { e = (e - mean_0); });
//    std::for_each(var_x.begin(), var_x.end(), [&](double& e) { e = (e - mean_x); });
//
//    vector<double> var_corre;
//    std::transform(var_0.begin(), var_0.end(), var_x.begin(), std::back_inserter(var_corre), std::multiplies<>());
//
//    std::for_each(var_0.begin(), var_0.end(), [&](double& e) { e = e * e; });
//    std::for_each(var_x.begin(), var_x.end(), [&](double& e) { e = e * e; });
//
//    std::for_each(sq_err.begin(), sq_err.end(), [&](double& e) { e = e * e; });
//
//    double sum_var_0  = std::accumulate(var_0.begin(), var_0.end(), 0.0);
//    double sum_var_x  = std::accumulate(var_x.begin(), var_x.end(), 0.0);
//    double sum_sq_err = std::accumulate(sq_err.begin(), sq_err.end(), 0.0);
//    double sum_corre  = std::accumulate(var_corre.begin(), var_corre.end(), 0.0);
//
//    vector<double> rel_abserr;
//    std::transform(err.begin(), err.end(), oData.begin(), std::back_inserter(rel_abserr), std::divides<>());
//    std::for_each(rel_abserr.begin(), rel_abserr.end(), [](double& e) { e = fabs(e); });
//    std::remove_if(rel_abserr.begin(), rel_abserr.end(), [](double& e) { return e == std::numeric_limits<double>::infinity(); });
//    double max_pwr_relerr = *std::max_element(rel_abserr.begin(), rel_abserr.end());
//
//    double std_0 = sqrt(sum_var_0 / __len);
//    double std_x = sqrt(sum_var_x / __len);
//    double ee    = sum_corre / __len;
//    double acEff = ee / std_0 / std_x;
//
//    double MSE   = sum_sq_err / __len;
//    double PSNR  = 20 * log10(val_range) - 10 * log10(MSE);
//    double NRMSE = sqrt(MSE) / val_range;
//
//    printf("min.value\t%.20G\n", min_val);
//    printf("max.value\t%.20G\n", max_val);
//    printf("val.range\t%.20G\n", val_range);
//    printf("max.abs.err\t%.10f\n", max_abserr);
//    printf("max.abs.rel.err\t%f\n", max_abserr / (max_val - min_val));
//    printf("max.pwr.abs.rel.err\t%f\n", max_pwr_relerr);
//    printf("P.S.N.R.\t%f\n", PSNR);
//    printf("N.R.M.S.E.\t%.20G\n", NRMSE);
//    printf("correl.coeff\t%f\n", acEff);
//    if (c_byte_size) printf("compress.ratio\t%f\n", 1.0 * __len * sizeof(T) / c_byte_size);
//}

namespace Analysis {
template <typename T>
void VerifyData(T*     xData,
                T*     oData,
                size_t _len,
                bool   override_eb       = false,
                double new_eb            = 0,
                size_t archive_byte_size = 0,
                size_t binning_scale     = 1) {
    double _max = 0, _min = 0, max_abserr = 0;
    _max = oData[0], _min = oData[0];
    max_abserr = xData[0] > oData[0] ? xData[0] - oData[0] : oData[0] - xData[0];

    double sum_0 = 0, sum_x = 0;

    for (size_t i = 0; i < _len; i++) sum_0 += oData[i], sum_x += xData[i];

    double mean_0 = sum_0 / _len;
    double mean_x = sum_x / _len;

    double sum_var_0  = 0;
    double sum_var_x  = 0;
    double sum_sq_err = 0;
    double sum_corre  = 0;
    double rel_abserr = 0;

    double max_pwr_rel_abserr = 0;
    int    max_abserr_index   = 0;
    for (size_t i = 0; i < _len; i++) {
        _max = _max < oData[i] ? oData[i] : _max;
        _min = _min > oData[i] ? oData[i] : _min;

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
    double coeff = ee / std_0 / std_x;

    double rng   = _max - _min;
    double MSE   = sum_sq_err / _len;
    double PSNR  = 20 * log10(rng) - 10 * log10(MSE);
    double NRMSE = sqrt(MSE) / rng;

    auto left_border  = []() { printf("| "); };
    auto right_border = []() { printf("\n"); };

    cout << endl;
    cout << log_info << "verification start ---------------------" << endl;
    left_border(), printf("%-20s%.20G", "min.val", _min), right_border();
    left_border(), printf("%-20s%.20G", "max.val", _max), right_border();
    left_border(), printf("%-20s%.20G", "val.rng", rng), right_border();
    left_border(), printf("%-20s\e[31m%.20G\e[0m", "max.err.abs.val", max_abserr), right_border();
    left_border(), printf("%-20s%d", "max.err.abs.idx", max_abserr_index), right_border();
    if (override_eb) {
        left_border(), printf("----------------------------------------------------------------\n");
        left_border(), printf("OVERRODE eb (because of, for example, binning) to:\t%.6G", new_eb), right_border();
        left_border(), printf("max.err.abs.val/OVERRIDEN eb:\t%.6G", max_abserr / new_eb), right_border();
        left_border(), printf("----------------------------------------------------------------\n");
    } else {
        left_border(), printf("%-20s\e[31m%.20G\e[0m", "max.err.vs.rng", max_abserr / (_max - _min)), right_border();
    }
    left_border(), printf("%-20s%.20G", "max.pw.rel.err", max_pwr_rel_abserr), right_border();
    left_border(), printf("%-20s\e[31m%.20G\e[0m", "PSNR", PSNR), right_border();
    left_border(), printf("%-20s%.20G", "NRMSE", NRMSE), right_border();
    left_border(), printf("%-20s%.20G", "correl.coeff", coeff), right_border();
    if (archive_byte_size) {
        left_border(), printf("%-20s\e[31m%lf\e[0m", "compression.ratio", binning_scale * 1.0 * _len * sizeof(T) / archive_byte_size), right_border();
    }
    cout << log_info << "verification end -----------------------" << endl;
    cout << endl;
}

/*
template <typename T>
void psnr(T* xData, T* oData, size_t _len, size_t c_byte_size = 0) {
    double _max = 0, _min = 0, max_abserr = 0;
    _max = oData[0], _min = oData[0];
    max_abserr = xData[0] > oData[0] ? xData[0] - oData[0] : oData[0] - xData[0];

    double sum_0 = 0, sum_x = 0;

    for (size_t i = 0; i < _len; i++) sum_0 += oData[i], sum_x += xData[i];

    double mean_0 = sum_0 / _len;
    double mean_x = sum_x / _len;

    double sum_var_0  = 0;
    double sum_var_x  = 0;
    double sum_sq_err = 0;
    double sum_corre  = 0;
    double rel_abserr = 0;

    double max_pwr_rel_abserr = 0;
    for (size_t i = 0; i < _len; i++) {
        _max = _max < oData[i] ? oData[i] : _max;
        _min = _min > oData[i] ? oData[i] : _min;

        float abserr = fabs(xData[i] - oData[i]);

        if (oData[i] != 0) {
            rel_abserr         = abserr / fabs(oData[i]);
            max_pwr_rel_abserr = max_pwr_rel_abserr < rel_abserr ? rel_abserr : max_pwr_rel_abserr;
        }
        max_abserr = max_abserr < abserr ? abserr : max_abserr;

        sum_corre += (oData[i] - mean_0) * (xData[i] - mean_x);
        sum_var_0 += (oData[i] - mean_0) * (oData[i] - mean_0);
        sum_var_x += (xData[i] - mean_x) * (xData[i] - mean_x);
        sum_sq_err += abserr * abserr;
    }
    double std_0 = sqrt(sum_var_0 / _len);
    double std_x = sqrt(sum_var_x / _len);
    double ee    = sum_corre / _len;
    double coeff = ee / std_0 / std_x;

    double rng   = _max - _min;
    double MSE   = sum_sq_err / _len;
    double PSNR  = 20 * log10(rng) - 10 * log10(MSE);
    double NRMSE = sqrt(MSE) / rng;

    //    printf("min.value:\t%.20G\n", __min);
    //    printf("max.value:\t%.20G\n", __max);
    //    printf("val.range:\t%.20G\n", rng);
    //    printf("\e[31mmax.abs.err:\t%.10f\e[0m\n", max_abserr);
    //    printf("\e[31mmax.rel.abs.err:\t%lf\e[0m\n", max_abserr / (__max - __min));
    //    printf("max.pw.rel.err:\t%lf\n", max_pwr_rel_abserr);
    printf("\e[31mPSNR:\t%f\e[0m\n", PSNR);
    printf("NRMSE:\t%.20G\n", NRMSE);
    printf("correl.coeff:\t%f\n", coeff);
    //    if (c_byte_size) printf("compress.ratio\t%f\n", 1.0 * __len * sizeof(T) / c_byte_size);
}
*/

}  // namespace Analysis

#endif
