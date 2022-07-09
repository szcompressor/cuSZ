/**
 * @file analysis.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.3
 * @date 2020-11-07
 *
 * (C) 2020 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef ANALYSIS_CUH
#define ANALYSIS_CUH

//  nvcc analysis.cu cuda_mem.o -std=c++14 -expt-extended-lambda -DMAIN -gencode arch=compute_75,code=sm_75

#define tix threadIdx.x

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <iostream>
#include <string>
#include <tuple>

#include "../common.hh"

namespace cusz {

static const int MINVAL = 0;
static const int MAXVAL = 1;
static const int AVGVAL = 2;
static const int RNG    = 3;

template <typename T>
void get_minmaxavg_rng(thrust::device_ptr<T> g_ptr, size_t len, T res[4])
{
    auto minel  = thrust::min_element(g_ptr, g_ptr + len) - g_ptr;  // excluding padded
    auto maxel  = thrust::max_element(g_ptr, g_ptr + len) - g_ptr;  // excluding padded
    res[MINVAL] = *(g_ptr + minel);
    res[MAXVAL] = *(g_ptr + maxel);
    res[RNG]    = res[MAXVAL] - res[MINVAL];

    auto sum    = thrust::reduce(g_ptr, g_ptr + len, (T)0.0, thrust::plus<T>());
    res[AVGVAL] = sum / len;
}

template <typename T>
void get_max_err(
    T*      reconstructed,  // in
    T*      original,       // in
    size_t  len,            // in
    T&      maximum_val,    // out
    size_t& maximum_loc,    // out
    bool    destructive = false)
{
    T* diff;

    if (destructive) {
        diff = original;  // aliasing
    }
    else {
        cudaMalloc(&diff, sizeof(T) * len);
    }

    auto expr = [=] __device__(T rel, T oel) { return rel - oel; };

    // typesafe (also with exec-policy binding)
    thrust::device_ptr<T> r(reconstructed);
    thrust::device_ptr<T> o(original);
    thrust::device_ptr<T> d(diff);

    thrust::transform(r, r + len, o, d, expr);

    auto maximum_ptr = thrust::max_element(d, d + len);
    maximum_val      = *maximum_ptr;
    maximum_loc      = maximum_ptr - d;

    if (not destructive) { cudaFree(diff); }
}

template <typename T>
void verify_data_GPU(cusz_stats* s, T* xdata, T* odata, size_t len)
{
    using tup = thrust::tuple<T, T>;

    thrust::device_ptr<T> p_odata = thrust::device_pointer_cast(odata);  // origin
    thrust::device_ptr<T> p_xdata = thrust::device_pointer_cast(xdata);

    T odata_res[4], xdata_res[4];

    get_minmaxavg_rng(p_odata, len, odata_res);
    get_minmaxavg_rng(p_xdata, len, xdata_res);

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(p_odata, p_xdata));
    auto end   = thrust::make_zip_iterator(thrust::make_tuple(p_odata + len, p_xdata + len));

    // clang-format off
    auto corr      = [=] __host__ __device__(tup t)  { return (thrust::get<0>(t) - odata[AVGVAL]) * (thrust::get<1>(t) - xdata[AVGVAL]); };
    auto err2      = []  __host__ __device__(tup t)  { T f = thrust::get<0>(t) - thrust::get<1>(t); return f * f; };
    auto var_odata = [=] __host__ __device__(T a) { T f = a - odata[AVGVAL]; return f * f; };
    auto var_xdata = [=] __host__ __device__(T a) { T f = a - xdata[AVGVAL]; return f * f; };

    auto sum_err2      = thrust::transform_reduce(begin, end, err2, 0.0f, thrust::plus<T>());
    auto sum_corr      = thrust::transform_reduce(begin, end, corr, 0.0f, thrust::plus<T>());
    auto sum_var_odata = thrust::transform_reduce(p_odata, p_odata + len, var_odata, 0.0f, thrust::plus<T>());
    auto sum_var_xdata = thrust::transform_reduce(p_xdata, p_xdata + len, var_xdata, 0.0f, thrust::plus<T>());
    // clang-format on

    double std_odata = sqrt(sum_var_odata / len);
    double std_xdata = sqrt(sum_var_xdata / len);
    double ee        = sum_corr / len;

    // -----------------------------------------------------------------------------
    T      max_abserr{0};
    size_t max_abserr_index{0};
    get_max_err(xdata, odata, len, max_abserr, max_abserr_index, false);
    // -----------------------------------------------------------------------------

    s->len = len;

    s->odata.max = odata_res[MAXVAL];
    s->odata.min = odata_res[MINVAL];
    s->odata.rng = odata_res[MAXVAL] - odata_res[MINVAL];
    s->odata.std = std_odata;

    s->xdata.max = xdata_res[MAXVAL];
    s->xdata.min = xdata_res[MINVAL];
    s->xdata.rng = xdata_res[MAXVAL] - xdata_res[MINVAL];
    s->xdata.std = std_xdata;

    s->max_err.idx    = max_abserr_index;
    s->max_err.abs    = max_abserr;
    s->max_err.rel    = max_abserr / s->odata.rng;
    s->max_err.pwrrel = NAN;

    s->reduced.coeff = ee / std_odata / std_xdata;
    s->reduced.MSE   = sum_err2 / len;
    s->reduced.NRMSE = sqrt(s->reduced.MSE) / s->odata.rng;
    s->reduced.PSNR  = 20 * log10(s->odata.rng) - 10 * log10(s->reduced.MSE);
}

}  // namespace cusz

#endif
