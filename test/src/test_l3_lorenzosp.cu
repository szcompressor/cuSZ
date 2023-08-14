/**
 * @file test_l3_lorenzosp.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-04-05
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <cstdint>
#include <stdexcept>
#include <typeinfo>
#include "kernel/l23.hh"
#include "kernel/spv_gpu.hh"
#include "kernel/l23r.hh"
#include "mem/memseg_cxx.hh"
#include "mem/compact_cu.hh"
#include "rand.hh"
#include "stat/compare_cpu.hh"
#include "stat/compare_gpu.hh"
#include "utils/print_gpu.hh"
#include "utils/viewer.hh"

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

std::string type_literal;

template <typename T, typename EQ, bool LENIENT = true>
bool testcase(size_t const x, size_t const y, size_t const z, double const eb, int const radius = 512)
{
    // When the input type is FP<X>, the internal precision should be the same.
    using FP = T;
    auto len = x * y * z;

    auto oridata     = new pszmem_cxx<T>(x, y, z, "oridata");
    auto de_data     = new pszmem_cxx<T>(x, y, z, "de_data");
    auto outlier     = new pszmem_cxx<T>(x, y, z, "outlier, normal");
    auto ectrl_focus = new pszmem_cxx<EQ>(x, y, z, "ectrl_focus");
    auto ectrl_ref   = new pszmem_cxx<EQ>(x, y, z, "ectrl_ref");
    auto spval       = new pszmem_cxx<T>(x, y, z, "spval");
    auto spidx       = new pszmem_cxx<uint32_t>(x, y, z, "spidx");

    oridata->control({Malloc, MallocHost});
    de_data->control({Malloc, MallocHost});
    outlier->control({Malloc, MallocHost});
    ectrl_focus->control({Malloc, MallocHost});
    ectrl_ref->control({Malloc, MallocHost});
    spval->control({Malloc, MallocHost});
    spidx->control({Malloc, MallocHost});

    // setup dummy outliers
    auto ratio_outlier      = 0.00001;
    auto num_of_exaggerated = (int)(ratio_outlier * len);
    auto step               = (int)(len / (num_of_exaggerated + 1));

    cout << "num_of_exaggerated: " << num_of_exaggerated << endl;
    cout << "step of inserting outlier: " << step << endl;

    CompactCudaDram<T> compact_outlier;
    compact_outlier.reserve_space(len).control({Malloc, MallocHost});

    psz::testutils::cuda::rand_array<T>(oridata->dptr(), len);

    oridata->control({D2H});
    for (auto i = 0; i < num_of_exaggerated; i++) { oridata->hptr(i * step) *= 4; }
    oridata->control({H2D});

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float time;
    auto  len3 = dim3(x, y, z);

    psz_comp_l23r<T, EQ, false>(  //
        oridata->dptr(), len3, eb, radius, ectrl_focus->dptr(), &compact_outlier, &time, stream);
    cudaStreamSynchronize(stream);

    psz_comp_l23<T, EQ>(  //
        oridata->dptr(), len3, eb, radius, ectrl_ref->dptr(), outlier->dptr(), &time, stream);
    cudaStreamSynchronize(stream);

    ectrl_focus->control({ASYNC_D2H}, stream);
    ectrl_ref->control({ASYNC_D2H}, stream);
    cudaStreamSynchronize(stream);

    auto two_ectrl_eq = true;
    for (auto i = 0; i < len; i++) {
        auto e1 = ectrl_focus->hptr(i);
        auto e2 = ectrl_ref->hptr(i);
        if (e1 != e2) {
            printf("i: %d\t not equal\te1: %u\te2: %u\n", i, e1, e2);
            two_ectrl_eq = false;
        }
    }
    printf("    two kinds of ectrls equal?: %s\n", two_ectrl_eq ? "yes" : "no");

    compact_outlier.make_host_accessible(stream);

    ////////////////////////////////////////////////////////////////////////////////
    int splen;
    {
        thrust::sort_by_key(
            thrust::host,                                                          // execution
            compact_outlier.h_idx, compact_outlier.h_idx + compact_outlier.h_num,  // key
            compact_outlier.h_val);                                                // value

        float __t;
        psz::spv_gather<T, uint32_t>(outlier->dptr(), len, spval->dptr(), spidx->dptr(), &splen, &__t, stream);
        spidx->control({D2H});
        spval->control({D2H});

        auto two_outlier_eq = true;

        for (auto i = 0; i < splen; i++) {
            auto normal_idx  = spidx->hptr(i);
            auto normal_val  = spval->hptr(i);
            auto compact_idx = compact_outlier.h_idx[i];
            auto compact_val = compact_outlier.h_val[i];

            if (normal_idx != compact_idx or normal_val != compact_val) {
                // printf(
                //     "i: %d\t"
                //     "normal-(idx,val)=(%u,%4.1f)\t"
                //     "compact-(idx,val)=(%u,%4.1f)"
                //     "\n",
                //     i, normal_idx, normal_val, compact_idx, compact_val);
                two_outlier_eq = false;
            }
        }
        printf("#normal_outlier: %d\n", splen);
        printf("#compact_outlier: %d\n", compact_outlier.num_outliers());

        printf("    two kinds of outliers equal?: %s\n", two_outlier_eq ? "yes" : "no");
    }
    // //\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


    psz_adhoc_scttr(
        compact_outlier.val(), compact_outlier.idx(), compact_outlier.num_outliers(), de_data->dptr(), &time, stream);

    psz_decomp_l23<T, EQ, FP>(
        ectrl_focus->dptr(), len3, de_data->dptr(), eb, radius,
        de_data->dptr(),  //
        &time, stream);
    cudaStreamSynchronize(stream);

    de_data->control({D2H});

    size_t first_non_eb = 0;
    // bool   error_bounded = psz::thrustgpu_error_bounded<T>(de_data, oridata, len, eb, &first_non_eb);
    bool error_bounded = psz::cppstd_error_bounded<T>(de_data->hptr(), oridata->hptr(), len, eb, &first_non_eb);

    // psz::eval_dataquality_gpu(oridata->dptr(), de_data->dptr(), len);

    cudaStreamDestroy(stream);

    delete oridata;
    delete de_data;
    delete ectrl_focus;
    delete ectrl_ref;
    delete outlier;
    delete spidx;
    delete spval;

    compact_outlier.control({Free, FreeHost});

    printf("(%u,%u,%u)\t(T=%s,EQ=%s)\terror bounded?\t", x, y, z, typeid(T).name(), typeid(EQ).name());
    if (not LENIENT) {
        if (not error_bounded) throw std::runtime_error("NO");
    }
    else {
        cout << (error_bounded ? "yes" : "NO") << endl;
    }

    return error_bounded;
}

bool batch_run_testcase(uint32_t x, uint32_t y, uint32_t z)
{
    auto all_pass = true;

    auto ndim = [&]() -> int {
        auto _ndim = 3;
        if (z == 1) _ndim = 2;
        if (y == 1) _ndim = 1;
        if (x == 1) throw std::runtime_error("x cannot be 1");
        return _ndim;
    };

    double eb = 1e-3;  // for 1D
    if (ndim() == 2) eb = 3e-3;
    if (ndim() == 3) eb = 3e-3;

    // all_pass = all_pass and testcase<float, uint8_t>(x, y, z, eb, 128);
    // all_pass = all_pass and testcase<float, uint16_t>(x, y, z, eb, 512);
    all_pass = all_pass and testcase<float, uint32_t>(x, y, z, eb, 512);
    // all_pass = all_pass and testcase<float, float>(x, y, z, eb, 512);
    // all_pass = all_pass and testcase<double, uint8_t>(x, y, z, eb, 128);
    // all_pass = all_pass and testcase<double, uint16_t>(x, y, z, eb, 512);
    // all_pass = all_pass and testcase<double, uint32_t>(x, y, z, eb, 512);
    // all_pass = all_pass and testcase<double, float>(x, y, z, eb, 512);

    // all_pass = all_pass and testcase<float, int32_t>(x, y, z, eb, 512);
    // all_pass = all_pass and testcase<double, int32_t>(x, y, z, eb, 512);

    return all_pass;
}

int main(int argc, char** argv)
{
    cudaDeviceReset();

    bool all_pass = true;

    all_pass = all_pass and batch_run_testcase(6480000, 1, 1);
    printf("\n");
    all_pass = all_pass and batch_run_testcase(3600, 1800, 1);
    printf("\n");
    all_pass = all_pass and batch_run_testcase(360, 180, 100);

    if (all_pass)
        return 0;
    else
        return -1;
}
