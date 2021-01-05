/**
 * @file analysis.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.1.3
 * @date 2020-11-07
 *
 * (C) 2020 by Washington State University, Argonne National Laboratory
 *
 */

//  nvcc analysis.cu cuda_mem.o -std=c++14 -expt-extended-lambda -DMAIN -gencode arch=compute_75,code=sm_75

#define tix threadIdx.x

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <iostream>
#include <string>
#include <tuple>

#include "analysis.cuh"
#include "utils/cuda_mem.cuh"

#if __cplusplus >= 201402L

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

    std::cout << min_val << std::endl;
    std::cout << max_val << std::endl;
    std::cout << rng << std::endl;
    std::cout << mean << std::endl;
    // TODO redundant types
    return std::make_tuple<Data, Data, Data, Data>((Data)min_val, (Data)max_val, (Data)rng, (Data)mean);
}

template <typename Data>
void GetPSNR(Data* x, Data* y, size_t len)
{
    using tup = thrust::tuple<Data, Data>;

    thrust::device_ptr<Data> x_ptr = thrust::device_pointer_cast(x);  // origin
    thrust::device_ptr<Data> y_ptr = thrust::device_pointer_cast(y);

    Data x_min_val, x_max_val, x_rng, mean_x;
    Data y_min_val, y_max_val, y_rng, mean_y;
    std::tie(x_min_val, x_max_val, x_rng, mean_x) = GetMinMaxRng(x_ptr, len);
    std::tie(y_min_val, y_max_val, y_rng, mean_y) = GetMinMaxRng(y_ptr, len);

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(x_ptr, y_ptr));
    auto end   = thrust::make_zip_iterator(thrust::make_tuple(x_ptr + len, y_ptr + len));

    // clang-format off
    auto corr = [=] __host__ __device__(tup t)  { return (thrust::get<0>(t) - mean_x) * (thrust::get<1>(t) - mean_y); };
    auto err2 = []  __host__ __device__(tup t)  { Data f = thrust::get<0>(t) - thrust::get<1>(t); return f * f; };
    auto varx = [=] __host__ __device__(Data a) { Data f = a - mean_x; return f * f; };
    auto vary = [=] __host__ __device__(Data a) { Data f = a - mean_y; return f * f; };

    auto sum_err2 = thrust::transform_reduce(begin, end, err2, 0.0f, thrust::plus<Data>());
    auto sum_corr = thrust::transform_reduce(begin, end, corr, 0.0f, thrust::plus<Data>());
    auto sum_varx = thrust::transform_reduce(y_ptr, y_ptr + len, varx, 0.0f, thrust::plus<Data>());
    auto sum_vary = thrust::transform_reduce(y_ptr, y_ptr + len, vary, 0.0f, thrust::plus<Data>());
    // clang-format on

    double stdx = sqrt(sum_varx / len),                 //
        stdy    = sqrt(sum_vary / len),                 //
        ee      = sum_corr / len,                       //
        coeff   = ee / stdx / stdy,                     //
        MSE     = sum_err2 / len,                       //
        PSNR    = 20 * log10(x_rng) - 10 * log10(MSE),  //
        NRMSE   = sqrt(MSE) / x_rng;

    std::cout << "PSNR:\t" << PSNR << std::endl;
    std::cout << "coeff:\t" << coeff << std::endl;
    std::cout << "NRMSE:\t" << NRMSE << std::endl;
}

#ifdef MAIN
int main()
{
    // size_t len = 1168 * 1126 * 922;
    size_t len = 3600 * 1800;
    // auto   origin  = io::ReadBinaryFile<float>("/home/jtian/Parihaka_PSTM_far_stack.f32", len);
    // auto   extract = io::ReadBinaryFile<float>("/home/jtian/Parihaka_PSTM_far_stack.f32.szx", len);

    auto a       = hires::now();
    auto origin  = io::ReadBinaryToNewArray<float>("/home/jtian/cusz-rolling/data/sample-cesm-CLDHGH", len);
    auto extract = io::ReadBinaryToNewArray<float>("/home/jtian/cusz-rolling/data/sample-cesm-CLDHGH.szx", len);
    auto z       = hires::now();
    std::cout << "time loading data:\t" << static_cast<duration_t>(z - a).count() << endl;

    auto d_origin  = mem::CreateDeviceSpaceAndMemcpyFromHost(origin, len);
    auto d_extract = mem::CreateDeviceSpaceAndMemcpyFromHost(extract, len);

    // thrust::device_ptr<float> origin_ptr  = thrust::device_pointer_cast(d_origin);   // origin
    // thrust::device_ptr<float> extract_ptr = thrust::device_pointer_cast(d_extract);  // origin
    // GetMinMaxRng(origin_ptr, len);
    // GetMinMaxRng(extract_ptr, len);

    auto aa = hires::now();
    GetPSNR<float>(d_origin, d_extract, len);
    auto zz = hires::now();
    std::cout << "time kernel:\t" << static_cast<duration_t>(zz - aa).count() << endl;

    return 0;
}
#endif

#endif