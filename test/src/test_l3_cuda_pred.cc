/**
 * @file test_l3_cuda_pred.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-06
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include <typeinfo>
#include "cli/quality_viewer.hh"
#include "kernel/cpplaunch_cuda.hh"
#include "kernel/lorenzo_all.hh"
#include "rand.hh"
#include "stat/compare_cpu.hh"
#include "stat/compare_gpu.hh"
#include "utils/print_gpu.hh"

std::string type_literal;

template <typename T, typename EQ, bool LENIENT = true>
bool f(size_t const x, size_t const y, size_t const z, double const error_bound, int const radius = 512)
{
    // When the input type is FP<X>, the internal precision should be the same.
    using FP = T;
    auto len = x * y * z;

    Capsule<T> data(x, y, z, "data"), xdata(x, y, z, "xdata"), outlier(x, y, z, "outlier");
    data.mallocmanaged(), xdata.mallocmanaged(), outlier.mallocmanaged();

    Capsule<EQ> eq(x, y, z, "eq");
    eq.mallocmanaged();

    uint32_t* outlier_idx{nullptr};

    parsz::testutils::cuda::rand_array<T>(data.uniptr(), len);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float time;

    compress_predict_lorenzo_i<T, EQ, FP>(                    //
        data.uniptr(), data.len3(), error_bound, radius,      // input and config
        eq.uniptr(), outlier.uniptr(), outlier_idx, nullptr,  // output
        &time, stream);
    cudaStreamSynchronize(stream);

    decompress_predict_lorenzo_i<T, EQ, FP>(                       //
        eq.uniptr(), eq.len3(), outlier.uniptr(), outlier_idx, 0,  // input
        error_bound, radius,                                       // input (config)
        xdata.uniptr(),                                            // output
        &time, stream);
    cudaStreamSynchronize(stream);

    // accsz::peek_device_data(data.uniptr(), 100);
    // accsz::peek_device_data(xdata.uniptr(), 100);

    size_t first_non_eb = 0;
    // bool   error_bounded = parsz::thrustgpu_error_bounded<T>(xdata, data, len, error_bound, &first_non_eb);
    bool error_bounded = parsz::cppstd_error_bounded<T>(xdata.uniptr(), data.uniptr(), len, error_bound, &first_non_eb);

    // /* perform evaluation */ cusz::QualityViewer::echo_metric_gpu(data.uniptr(), xdata.uniptr(), len);

    cudaStreamDestroy(stream);
    data.freemanaged(), xdata.freemanaged(), eq.freemanaged(), outlier.freemanaged();

    printf("(%u,%u,%u)\t(T=%s,EQ=%s)\terror bounded?\t", x, y, z, typeid(T).name(), typeid(EQ).name());
    if (not LENIENT) {
        if (not error_bounded) throw std::runtime_error("NO");
    }
    else {
        cout << (error_bounded ? "yes" : "NO") << endl;
    }

    return error_bounded;
}

bool g(uint32_t x, uint32_t y, uint32_t z)
{
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

    all_pass = all_pass and f<float, int32_t>(x, y, z, eb, 512);
    all_pass = all_pass and f<double, int32_t>(x, y, z, eb, 512);

    return all_pass;
}

int main(int argc, char** argv)
{
    bool all_pass = true;
    all_pass      = all_pass and g(6480000, 1, 1);
    all_pass      = all_pass and g(3600, 1800, 1);
    all_pass      = all_pass and g(360, 180, 100);

    if (all_pass)
        return 0;
    else
        return -1;
}
