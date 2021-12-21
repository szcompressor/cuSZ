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

#ifdef MAIN
#include "verify.hh"
#endif

#include "cuda_mem.cuh"

template <typename T>
std::tuple<T, T, T, T> GetMinMaxRng(thrust::device_ptr<T> g_ptr, size_t len)
{
    size_t min_el_loc = thrust::min_element(g_ptr, g_ptr + len) - g_ptr;  // excluding padded
    size_t max_el_loc = thrust::max_element(g_ptr, g_ptr + len) - g_ptr;  // excluding padded
    T      min_val    = *(g_ptr + min_el_loc);
    T      max_val    = *(g_ptr + max_el_loc);
    T      rng        = max_val - min_val;

    T sum  = thrust::reduce(g_ptr, g_ptr + len, (T)0.0, thrust::plus<T>());
    T mean = sum / len;

    // std::cout << "min.val:\t" << min_val << '\n';
    // std::cout << "max.val:\t" << max_val << '\n';
    // std::cout << "rng:\t" << rng << '\n';
    // std::cout << "mean:\t" << mean << '\n';
    // TODO redundant types
    return std::make_tuple<T, T, T, T>((T)min_val, (T)max_val, (T)rng, (T)mean);
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
void verify_data_GPU(stat_t* stat, T* xdata, T* odata, size_t len)
{
    using tup = thrust::tuple<T, T>;

    thrust::device_ptr<T> p_odata = thrust::device_pointer_cast(odata);  // origin
    thrust::device_ptr<T> p_xdata = thrust::device_pointer_cast(xdata);

    T min_odata, max_odata, rng_odata, mean_odata;
    T min_xdata, max_xdata, rng_xdata, mean_xdata;
    std::tie(min_odata, max_odata, rng_odata, mean_odata) = GetMinMaxRng(p_odata, len);
    std::tie(min_xdata, max_xdata, rng_xdata, mean_xdata) = GetMinMaxRng(p_xdata, len);

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(p_odata, p_xdata));
    auto end   = thrust::make_zip_iterator(thrust::make_tuple(p_odata + len, p_xdata + len));

    // clang-format off
    auto corr      = [=] __host__ __device__(tup t)  { return (thrust::get<0>(t) - mean_odata) * (thrust::get<1>(t) - mean_xdata); };
    auto err2      = []  __host__ __device__(tup t)  { T f = thrust::get<0>(t) - thrust::get<1>(t); return f * f; };
    auto var_odata = [=] __host__ __device__(T a) { T f = a - mean_odata; return f * f; };
    auto var_xdata = [=] __host__ __device__(T a) { T f = a - mean_xdata; return f * f; };

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
    stat->max_pwrrel_abserr = NAN;

    stat->coeff = ee / std_odata / std_xdata;
    stat->MSE   = sum_err2 / len;
    stat->NRMSE = sqrt(stat->MSE) / stat->rng_odata;
    stat->PSNR  = 20 * log10(stat->rng_odata) - 10 * log10(stat->MSE);
}

#ifdef MAIN
int main(int argc, char** argv)
{
    if (argc < 4) {
        cout << "./prog <len> <original> <reconstructed>" << endl;
        exit(0);
    }

    // size_t len = 1168 * 1126 * 922;
    // size_t len = 3600* 1800;

    size_t len = atoi(argv[1]);

    auto a             = hires::now();
    auto original      = io::read_binary_to_new_array<float>(std::string(argv[2]), len);
    auto reconstructed = io::read_binary_to_new_array<float>(std::string(argv[3]), len);
    auto z             = hires::now();
    std::cout << "time loading data:\t" << static_cast<duration_t>(z - a).count() << endl;

    auto d_origin  = mem::create_devspace_memcpy_h2d(original, len);
    auto d_extract = mem::create_devspace_memcpy_h2d(reconstructed, len);

    auto   aa = hires::now();
    stat_t stat_gpu;
    verify_data_GPU<float>(&stat_gpu, d_origin, d_extract, len);
    analysis::print_data_quality_metrics<float>(&stat_gpu, 0, false);
    auto zz = hires::now();
    std::cout << "time kernel:\t" << static_cast<duration_t>(zz - aa).count() << endl;

    return 0;
}
#endif

#endif
