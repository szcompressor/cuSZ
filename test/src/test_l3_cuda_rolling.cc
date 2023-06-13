/**
 * @file test_l2_cuda_rolling.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-04-05
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <typeinfo>
#include "kernel/l23.hh"
#include "kernel2/l23r.hh"
#include "pipeline/compact_cuda.inl"
#include "rand.hh"
#include "stat/compare_cpu.hh"
#include "stat/compare_gpu.hh"
#include "utils/print_gpu.hh"
#include "utils/viewer.hh"
#include "utils2/memseg_cxx.hh"

std::string type_literal;

template <typename T, typename EQ, bool LENIENT = true>
bool f(size_t const x, size_t const y, size_t const z, double const eb, int const radius = 512)
{
    // When the input type is FP<X>, the internal precision should be the same.
    using FP = T;
    auto len = x * y * z;

    auto oridata = new pszmem_cxx<T>(x, y, z, "oridata");
    auto de_data = new pszmem_cxx<T>(x, y, z, "de_data");
    auto errctrl = new pszmem_cxx<EQ>(x, y, z, "errctrl");
    oridata->control({MallocManaged});
    de_data->control({MallocManaged});
    errctrl->control({MallocManaged});

    CompactCudaDram<T> outlier;
    outlier.set_reserved_len(len).malloc().mallochost();

    psz::testutils::cuda::rand_array<T>(oridata->uniptr(), len);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float time;
    auto  len3 = dim3(x, y, z);

    psz_comp_l23r<T, false, EQ>(  //
        oridata->uniptr(), len3, eb, radius, errctrl->uniptr(), &outlier, &time, stream);
    cudaStreamSynchronize(stream);

    outlier.make_host_accessible(stream);

    printf("#outlier: %d\n", outlier.num_outliers());

    psz_adhoc_scttr(outlier.val, outlier.idx, outlier.num_outliers(), de_data->uniptr(), &time, stream);

    psz_decomp_l23<T, EQ, FP>(
        errctrl->uniptr(), len3, de_data->uniptr(), eb, radius,
        de_data->uniptr(),  //
        &time, stream);
    cudaStreamSynchronize(stream);

    // psz::peek_device_data(oridata->uniptr(), 100);
    // psz::peek_device_data(de_data->uniptr(), 100);

    size_t first_non_eb = 0;
    // bool   error_bounded = psz::thrustgpu_error_bounded<T>(de_data, oridata, len, eb, &first_non_eb);
    bool error_bounded = psz::cppstd_error_bounded<T>(de_data->uniptr(), oridata->uniptr(), len, eb, &first_non_eb);

    // /* perform evaluation */ cusz::QualityViewer::echo_metric_gpu(oridata->uniptr(), de_data->uniptr(), len);

    cudaStreamDestroy(stream);
    delete oridata;
    delete de_data;
    delete errctrl;

    outlier.free().freehost();

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
    double eb       = 1e-3;

    // all_pass = all_pass and f<float, uint8_t>(x, y, z, eb, 128);
    // all_pass = all_pass and f<float, uint16_t>(x, y, z, eb, 512);
    all_pass = all_pass and f<float, uint32_t>(x, y, z, eb, 512);
    // all_pass = all_pass and f<float, float>(x, y, z, eb, 512);
    // all_pass = all_pass and f<double, uint8_t>(x, y, z, eb, 128);
    // all_pass = all_pass and f<double, uint16_t>(x, y, z, eb, 512);
    all_pass = all_pass and f<double, uint32_t>(x, y, z, eb, 512);
    // all_pass = all_pass and f<double, float>(x, y, z, eb, 512);

    // all_pass = all_pass and f<float, int32_t>(x, y, z, eb, 512);
    // all_pass = all_pass and f<double, int32_t>(x, y, z, eb, 512);

    return all_pass;
}

int main(int argc, char** argv)
{
    bool all_pass = true;

    all_pass = all_pass and g(6480000, 1, 1);
    // all_pass      = all_pass and g(3600, 1800, 1);
    // all_pass      = all_pass and g(360, 180, 100);

    if (all_pass)
        return 0;
    else
        return -1;
}
