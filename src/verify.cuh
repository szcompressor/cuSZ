#ifndef VERIFY_CUH
#define VERIFY_CUH

#include <cooperative_groups.h>
#include <cstdio>
#include <limits>
#include <numeric>
#include <vector>

#include "types.hh"

using namespace std;

// https://stackoverflow.com/a/51549250/8740097
__device__ __forceinline__ float atomicMin(float* addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value)))
                       : __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__device__ __forceinline__ float atomicMax(float* addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
                       : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__device__ __forceinline__ double atomicMin(double* addr, double value) {
    double old;
    old = (value >= 0) ? __int_as_double(atomicMin((int*)addr, __double_as_int(value)))
                       : __uint_as_double(atomicMax((unsigned int*)addr, __double_as_uint(value)));

    return old;
}

__device__ __forceinline__ double atomicMax(double* addr, double value) {
    double old;
    old = (value >= 0) ? __int_as_double(atomicMax((int*)addr, __double_as_int(value)))
                       : __uint_as_double(atomicMin((unsigned int*)addr, __double_as_uint(value)));

    return old;
}

namespace Analysis {

double _dg_max        = 0.0;
double _dg_min        = 0.0;
double _dg_max_abserr = 0.0;

template <typename T>
__global__ void getValRng(T* d, size_t len) {
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= len) return;
    grid_group g = this_grid();
    atomicMax(&_dg_max, d[gid]);
    g.sync();
    atomicMin(&_dg_min, d[gid]);
    g.sync();
}

template <typename T>
__global__ void getErrAnalysis(T* d0, T* d1, size_t len) {}

template <typename T>
void verify_data(T* xData, T* oData, size_t __len, bool override_eb = false, double new_eb = 0, size_t c_byte_size = 0) {
    double __max = 0, __min = 0, max_abserr = 0;
    __max = oData[0], __min = oData[0];
    max_abserr = xData[0] > oData[0] ? xData[0] - oData[0] : oData[0] - xData[0];

    double sum_0 = 0, sum_x = 0;

    for (size_t i = 0; i < __len; i++) sum_0 += oData[i], sum_x += xData[i];

    double mean_0 = sum_0 / __len;
    double mean_x = sum_x / __len;

    double sum_var_0  = 0;
    double sum_var_x  = 0;
    double sum_sq_err = 0;
    double sum_corre  = 0;
    double rel_abserr = 0;

    double max_pwr_rel_abserr = 0;
    for (size_t i = 0; i < __len; i++) {
        __max = __max < oData[i] ? oData[i] : __max;
        __min = __min > oData[i] ? oData[i] : __min;

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
    double std_0 = sqrt(sum_var_0 / __len);
    double std_x = sqrt(sum_var_x / __len);
    double ee    = sum_corre / __len;
    double coeff = ee / std_0 / std_x;

    double rng   = __max - __min;
    double MSE   = sum_sq_err / __len;
    double PSNR  = 20 * log10(rng) - 10 * log10(MSE);
    double NRMSE = sqrt(MSE) / rng;

    printf("%-20s%.20G\n", "min.val", __min);
    printf("%-20s%.20G\n", "max.val", __max);
    printf("%-20s%.20G\n", "val.rng", rng);
    // printf("max.val:\t%.20G\n", __max);
    // printf("val.rng:\t%.20G\n", rng);
    printf("%-20s\e[46m%.20G\e[0m\n", "max.err.abs.val", max_abserr);
    // printf("\e[46mmax.err.abs.val:\t%.10f\e[0m\n", max_abserr);
    if (override_eb) {
        printf("----------------------------------------------------------------\n");
        printf("OVERRODE eb (because of, for example, binning) to:\t%.6G\n", new_eb);
        printf("max.err.abs.val/OVERRIDEN eb:\t%.6G\n", max_abserr / new_eb);
        printf("----------------------------------------------------------------\n");
    } else {
        printf("%-20s\e[46m%.20G\e[0m\n", "max.err.vs.rng", max_abserr / (__max - __min));
    }
    printf("%-20s%.20G\n", "max.pw.rel.err", max_pwr_rel_abserr);
    printf("%-20s\e[1m%.20G\e[0m\n", "PSNR", PSNR);
    printf("%-20s%.20G\n", "NRMSE", NRMSE);
    printf("%-20s%.20G\n", "correl.coeff", coeff);
    if (c_byte_size) printf("%-20s\e[46m%lf\e[0m\n", "compression.ratio", 4.0 * __len * sizeof(T) / c_byte_size);
}

/*
template <typename T>
void psnr(T* xData, T* oData, size_t __len, size_t c_byte_size = 0) {
    double __max = 0, __min = 0, max_abserr = 0;
    __max = oData[0], __min = oData[0];
    max_abserr = xData[0] > oData[0] ? xData[0] - oData[0] : oData[0] - xData[0];

    double sum_0 = 0, sum_x = 0;

    for (size_t i = 0; i < __len; i++) sum_0 += oData[i], sum_x += xData[i];

    double mean_0 = sum_0 / __len;
    double mean_x = sum_x / __len;

    double sum_var_0  = 0;
    double sum_var_x  = 0;
    double sum_sq_err = 0;
    double sum_corre  = 0;
    double rel_abserr = 0;

    double max_pwr_rel_abserr = 0;
    for (size_t i = 0; i < __len; i++) {
        __max = __max < oData[i] ? oData[i] : __max;
        __min = __min > oData[i] ? oData[i] : __min;

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
    double std_0 = sqrt(sum_var_0 / __len);
    double std_x = sqrt(sum_var_x / __len);
    double ee    = sum_corre / __len;
    double coeff = ee / std_0 / std_x;

    double rng   = __max - __min;
    double MSE   = sum_sq_err / __len;
    double PSNR  = 20 * log10(rng) - 10 * log10(MSE);
    double NRMSE = sqrt(MSE) / rng;

    //    printf("min.value:\t%.20G\n", __min);
    //    printf("max.value:\t%.20G\n", __max);
    //    printf("val.range:\t%.20G\n", rng);
    //    printf("\e[46mmax.abs.err:\t%.10f\e[0m\n", max_abserr);
    //    printf("\e[46mmax.rel.abs.err:\t%lf\e[0m\n", max_abserr / (__max - __min));
    //    printf("max.pw.rel.err:\t%lf\n", max_pwr_rel_abserr);
    printf("\e[46mPSNR:\t%f\e[0m\n", PSNR);
    printf("NRMSE:\t%.20G\n", NRMSE);
    printf("correl.coeff:\t%f\n", coeff);
    //    if (c_byte_size) printf("compress.ratio\t%f\n", 1.0 * __len * sizeof(T) / c_byte_size);
}
*/

}  // namespace Analysis

#endif
