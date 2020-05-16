
#include <cstdio>
#include <limits>
#include <numeric>
#include <vector>

#include "format.hh"
#include "types.hh"
#include "verify.hh"

using namespace std;

template <typename T>
void analysis::VerifyData(T* xData, T* oData, size_t _len, bool override_eb, double new_eb, size_t archive_byte_size, size_t binning_scale)
{
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
    }
    else {
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

template void analysis::VerifyData<float>(float*, float*, size_t, bool, double, size_t, size_t);
/*
template void analysis::VerifyData<double>(double*, double*, size_t, bool, double, size_t, size_t);
template void analysis::VerifyData<char>(char*, char*, size_t, bool, double, size_t, size_t);
template void analysis::VerifyData<short>(short*, short*, size_t, bool, double, size_t, size_t);
template void analysis::VerifyData<int>(int*, int*, size_t, bool, double, size_t, size_t);
template void analysis::VerifyData<long>(long*, long*, size_t, bool, double, size_t, size_t);
template void analysis::VerifyData<long long>(long long*, long long*, size_t, bool, double, size_t, size_t);
 */
