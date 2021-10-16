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

template <typename Data>
std::tuple<Data, Data, Data, Data> GetMinMaxRng(thrust::device_ptr<Data> g_ptr, size_t len)
{
    size_t min_el_loc = thrust::min_element(g_ptr, g_ptr + len) - g_ptr;  // excluding padded
    size_t max_el_loc = thrust::max_element(g_ptr, g_ptr + len) - g_ptr;  // excluding padded
    Data   min_val    = *(g_ptr + min_el_loc);
    Data   max_val    = *(g_ptr + max_el_loc);
    Data   rng        = max_val - min_val;

    Data sum  = thrust::reduce(g_ptr, g_ptr + len, (Data)0.0, thrust::plus<Data>());
    Data mean = sum / len;

    // std::cout << "min.val:\t" << min_val << '\n';
    // std::cout << "max.val:\t" << max_val << '\n';
    // std::cout << "rng:\t" << rng << '\n';
    // std::cout << "mean:\t" << mean << '\n';
    // TODO redundant types
    return std::make_tuple<Data, Data, Data, Data>((Data)min_val, (Data)max_val, (Data)rng, (Data)mean);
}

template <typename Data>
void verify_data_GPU(stat_t* stat, Data* xdata, Data* odata, size_t len)
{
    using tup = thrust::tuple<Data, Data>;

    thrust::device_ptr<Data> p_odata = thrust::device_pointer_cast(odata);  // origin
    thrust::device_ptr<Data> p_xdata = thrust::device_pointer_cast(xdata);

    Data min_odata, max_odata, rng_odata, mean_odata;
    Data min_xdata, max_xdata, rng_xdata, mean_xdata;
    std::tie(min_odata, max_odata, rng_odata, mean_odata) = GetMinMaxRng(p_odata, len);
    std::tie(min_xdata, max_xdata, rng_xdata, mean_xdata) = GetMinMaxRng(p_xdata, len);

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(p_odata, p_xdata));
    auto end   = thrust::make_zip_iterator(thrust::make_tuple(p_odata + len, p_xdata + len));

    // clang-format off
    auto corr      = [=] __host__ __device__(tup t)  { return (thrust::get<0>(t) - mean_odata) * (thrust::get<1>(t) - mean_xdata); };
    auto err2      = []  __host__ __device__(tup t)  { Data f = thrust::get<0>(t) - thrust::get<1>(t); return f * f; };
    auto var_odata = [=] __host__ __device__(Data a) { Data f = a - mean_odata; return f * f; };
    auto var_xdata = [=] __host__ __device__(Data a) { Data f = a - mean_xdata; return f * f; };

    auto sum_err2      = thrust::transform_reduce(begin, end, err2, 0.0f, thrust::plus<Data>());
    auto sum_corr      = thrust::transform_reduce(begin, end, corr, 0.0f, thrust::plus<Data>());
    auto sum_var_odata = thrust::transform_reduce(p_odata, p_odata + len, var_odata, 0.0f, thrust::plus<Data>());
    auto sum_var_xdata = thrust::transform_reduce(p_xdata, p_xdata + len, var_xdata, 0.0f, thrust::plus<Data>());
    // clang-format on

    double std_odata = sqrt(sum_var_odata / len);
    double std_xdata = sqrt(sum_var_xdata / len);
    double ee        = sum_corr / len;

    stat->len       = len;
    stat->max_odata = max_odata;
    stat->min_odata = min_odata;
    stat->rng_odata = max_odata - min_odata;
    stat->std_odata = std_odata;
    stat->max_xdata = max_xdata;
    stat->min_xdata = min_xdata;
    stat->rng_xdata = max_xdata - min_xdata;
    stat->std_xdata = std_xdata;
    stat->coeff     = ee / std_odata / std_xdata;
    // stat->max_abserr_index  = max_abserr_index;
    // stat->max_abserr        = max_abserr;
    // stat->max_abserr_vs_rng = max_abserr / stat->rng_odata;
    // stat->max_pwrrel_abserr = max_pwrrel_abserr;
    stat->MSE   = sum_err2 / len;
    stat->NRMSE = sqrt(stat->MSE) / stat->rng_odata;
    stat->PSNR  = 20 * log10(stat->rng_odata) - 10 * log10(stat->MSE);
}

#ifdef MAIN
int main()
{
    // size_t len = 1168 * 1126 * 922;
    size_t len = 3600 * 1800;
    // auto   origin  = io::ReadBinaryFile<float>("/home/jtian/Parihaka_PSTM_far_stack.f32", len);
    // auto   extract = io::ReadBinaryFile<float>("/home/jtian/Parihaka_PSTM_far_stack.f32.cuszx", len);

    auto a       = hires::now();
    auto origin  = io::read_binary_to_new_array<float>("/home/jtian/cusz-rolling/data/sample-cesm-CLDHGH", len);
    auto extract = io::read_binary_to_new_array<float>("/home/jtian/cusz-rolling/data/sample-cesm-CLDHGH-3600x1800.cuszx", len);
    auto z       = hires::now();
    std::cout << "time loading data:\t" << static_cast<duration_t>(z - a).count() << endl;

    auto d_origin  = mem::create_devspace_memcpy_h2d(origin, len);
    auto d_extract = mem::create_devspace_memcpy_h2d(extract, len);

    // thrust::device_ptr<float> origin_ptr  = thrust::device_pointer_cast(d_origin);   // origin
    // thrust::device_ptr<float> extract_ptr = thrust::device_pointer_cast(d_extract);  // origin
    // GetMinMaxRng(origin_ptr, len);
    // GetMinMaxRng(extract_ptr, len);

    auto   aa = hires::now();
    stat_t stat_gpu;
    verify_data_GPU<float>(&stat_gpu, d_origin, d_extract, len);
    analysis::print_data_quality_metrics<Data>(&stat_gpu, 0, false );
    auto zz = hires::now();
    std::cout << "time kernel:\t" << static_cast<duration_t>(zz - aa).count() << endl;

    return 0;
}
#endif

#endif
