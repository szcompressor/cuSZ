/**
 * @file pred_ll.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-06
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "cli/quality_viewer.hh"
#include "kernel/cpplaunch_cuda.hh"
#include "kernel/lorenzo_all.hh"
#include "rand.hh"
#include "stat/compare_cpu.hh"
#include "stat/compare_gpu.hh"
#include "utils/print_gpu.hh"

std::string type_literal;

template <typename T, typename E>
bool f(size_t const x, size_t const y, size_t const z, double const error_bound, int const radius = 512)
{
    // When the input type is FP<X>, the internal precision should be the same.
    using FP = T;
    auto len = x * y * z;

    T *       data{nullptr}, *xdata{nullptr}, *outlier{nullptr}, *anchor{nullptr};
    uint32_t* outlier_idx{nullptr};
    E*        eq;
    dim3      len3       = dim3(x, y, z);
    dim3      dummy_len3 = dim3(0, 0, 0);

    cudaMallocManaged(&data, sizeof(T) * len);
    cudaMallocManaged(&xdata, sizeof(T) * len);
    cudaMallocManaged(&outlier, sizeof(T) * len);
    cudaMallocManaged(&eq, sizeof(E) * len);

    parsz::testutils::cuda::rand_array<T>(data, len);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float time;

    compress_predict_lorenzo_i<T, E, FP>(  //
        data, len3, error_bound, radius,   // input and config
        eq, dummy_len3, anchor, dummy_len3, outlier, outlier_idx,
        nullptr,  // output
        &time, stream);
    cudaStreamSynchronize(stream);

    decompress_predict_lorenzo_i<T, E, FP>(                           //
        eq, dummy_len3, anchor, dummy_len3, outlier, outlier_idx, 0,  // input
        error_bound, radius,                                          // input (config)
        xdata, len3,                                                  // output
        &time, stream);
    cudaStreamSynchronize(stream);

    accsz::peek_device_data(data, 100);
    accsz::peek_device_data(xdata, 100);

    size_t first_non_eb = 0;
    // bool   error_bounded = parsz::thrustgpu_error_bounded<T>(xdata, data, len, error_bound, &first_non_eb);
    bool error_bounded = parsz::cppstd_error_bounded<T>(xdata, data, len, error_bound, &first_non_eb);

    printf("error bound:\t%d\n", (int)error_bounded);
    // printf("first non error bounded:\t%lu\n", first_non_eb);

    /* perform evaluation */ cusz::QualityViewer::echo_metric_gpu(data, xdata, len);

    cudaStreamDestroy(stream);
    cudaFree(data), cudaFree(xdata), cudaFree(eq), cudaFree(outlier);

    if (not error_bounded) throw std::runtime_error("not error bounded");

    return error_bounded;
}

int main(int argc, char** argv)
{
    auto x = 360, y = 180, z = 100;

    auto   all_pass = true;
    double eb       = 1e-4;

    all_pass = all_pass and f<float, uint8_t>(x, y, z, eb, 128);
    all_pass = all_pass and f<float, uint16_t>(x, y, z, eb, 512);
    all_pass = all_pass and f<float, uint32_t>(x, y, z, eb, 512);
    all_pass = all_pass and f<float, float>(x, y, z, eb, 512);
    all_pass = all_pass and f<double, uint8_t>(x, y, z, eb, 128);
    all_pass = all_pass and f<double, uint16_t>(x, y, z, eb, 512);
    all_pass = all_pass and f<double, uint32_t>(x, y, z, eb, 512);
    all_pass = all_pass and f<double, float>(x, y, z, eb, 512);

    if (all_pass)
        return 0;
    else
        return -1;
}
