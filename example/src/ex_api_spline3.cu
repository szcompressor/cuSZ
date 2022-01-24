/**
 * @file ex_api_spline3.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-06-06
 * (create) 2021-06-06 (rev) 2022-01-08
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "ex_common.cuh"
#include "ex_common2.cuh"

#include "sp_path.cuh"
#include "wrapper/csr11.cuh"
#include "wrapper/interp_spline3.cuh"

#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

using std::cout;

using T = float;
using E = float;

// double const  eb          = 3e-3;
constexpr int fake_radius = 0;
// constexpr auto DEVICE      = cusz::LOC::DEVICE;

constexpr unsigned int dimx = 235, dimy = 449, dimz = 449;
constexpr dim3         xyz = dim3(dimx, dimy, dimz);
constexpr unsigned int len = dimx * dimy * dimz;

using Compressor = SparsityAwarePath::DefaultCompressor;
using Predictor  = cusz::Spline3<T, E, float>;
using SpReducer  = cusz::CSR11<T>;

std::string fname("");

template <typename T>
void echo_metric(T* d1, T* d2)
{
    stat_t stat;
    verify_data_GPU<T>(&stat, d1, d2, len);
    analysis::print_data_quality_metrics<T>(&stat, 0, false);
}

void predictor_detail(T* data, T* cmp, dim3 xyz, double eb, bool use_sp, cudaStream_t stream = nullptr)
{
    auto BARRIER = [&]() {
        if (not stream) {
            CHECK_CUDA(cudaDeviceSynchronize());
            printf("device sync'ed\n");
        }
        else {
            CHECK_CUDA(cudaStreamSynchronize(stream));
            printf("stream sync'ed\n");
        }
    };

    Predictor predictor(xyz, true);
    SpReducer spreducer;

    T* xdata = data;

    T* anchor{nullptr};
    E* errctrl{nullptr};

    auto dbg_echo_nnz = [&]() {
        int __nnz = thrust::count_if(
            thrust::device, errctrl, errctrl + predictor.get_quant_footprint(),
            [] __device__(const T& x) { return x != 0; });
        cout << "__nnz: " << __nnz << '\n';
    };

    BYTE*  csr;
    size_t csr_nbyte;

    auto _1_allocate_workspace = [&]() {  //
        printf("_1_allocate_workspace\n");
        predictor.allocate_workspace();
    };

    auto _1_compress_time = [&]() {
        printf("_1_compress_time\n");
        predictor.construct(data, eb, fake_radius, anchor, errctrl, stream);
        BARRIER();

        dbg_echo_nnz();
    };

    auto _1_decompress_time = [&]() {  //
        printf("_1_decompress_time\n");
        predictor.reconstruct(anchor, errctrl, eb, fake_radius, xdata, stream);
        BARRIER();
    };

    auto _2_allocate_workspace = [&]() {
        printf("_2_allocate_workspace\n");
        predictor.allocate_workspace();
        auto spreducer_in_len = predictor.get_quant_footprint();
        spreducer.allocate_workspace(spreducer_in_len, true);
    };

    auto _2_compress_time = [&]() {
        printf("_2_compress_time\n");
        predictor.construct(data, eb, fake_radius, anchor, errctrl, stream);
        BARRIER();

        dbg_echo_nnz();

        spreducer.gather_new(errctrl, predictor.get_quant_footprint(), csr, csr_nbyte, stream);
        BARRIER();
    };

    auto _2_decompress_time = [&]() {  //
        printf("_2_decompress_time\n");
        spreducer.scatter_new(csr, errctrl, stream);
        BARRIER();
        predictor.reconstruct(anchor, errctrl, eb, fake_radius, xdata, stream);
        BARRIER();
    };

    // -----------------------------------------------------------------------------

    if (not use_sp) {
        _1_allocate_workspace();
        _1_compress_time();
        _1_decompress_time();
    }
    else {
        _2_allocate_workspace();
        _2_compress_time();
        _2_decompress_time();
    }

    echo_metric(xdata, cmp);
}

void compressor_detail(T* data, T* cmp, dim3 xyz, double eb, bool use_sp, cudaStream_t stream = nullptr)
{
    Compressor compressor;
    BYTE*      compressed;
    size_t     compressed_len;

    auto xdata = data;

    compressor.allocate_workspace(xyz);
    compressor.compress(data, eb, fake_radius, compressed, compressed_len, stream);

    Capsule<BYTE> file(compressed_len);
    file.template alloc<cusz::LOC::HOST_DEVICE>();
    cudaMemcpy(file.dptr, compressed, compressed_len, cudaMemcpyDeviceToDevice);
    cudaMemset(compressed, 0x0, compressed_len);

    cudaMemcpy(compressed, file.dptr, compressed_len, cudaMemcpyDeviceToDevice);
    compressor.decompress(compressed, eb, fake_radius, xdata, stream);

    echo_metric(xdata, cmp);
}

void predictor_demo(bool use_sp, double eb = 1e-2, bool use_compressor = false, bool use_r2r = false)
{
    Capsule<T> exp(len, "exp data");
    Capsule<T> bak(len, "bak data");

    cout << "using eb = " << eb << '\n';
    cout << fname << '\n';

    exp.template alloc<cusz::LOC::HOST_DEVICE>().template from_file<cusz::LOC::HOST>(fname).host2device();
    bak.template alloc<cusz::LOC::HOST_DEVICE>();
    cudaMemcpy(bak.hptr, exp.hptr, len * sizeof(T), cudaMemcpyHostToHost);
    bak.host2device();

    double adjusted_eb;
    figure_out_eb(exp, eb, adjusted_eb, use_r2r);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    if (not use_compressor) {  //
        predictor_detail(exp.dptr, bak.dptr, xyz, adjusted_eb, use_sp, stream);
    }
    else {
        compressor_detail(exp.dptr, bak.dptr, xyz, adjusted_eb, use_sp, stream);
    }
    if (stream) CHECK_CUDA(cudaStreamDestroy(stream));

    exp.template free<cusz::LOC::HOST_DEVICE>();
    bak.template free<cusz::LOC::HOST_DEVICE>();
}

int main(int argc, char** argv)
{
    auto help = []() {
        cout << "./prog (1|2|3) fname [eb = 1e-2] [abs|r2r]\n";
        cout << "(1) predictor demo\n"
                "(2) predictor-spreducer demo\n"
                "(3) compressor integration demo\n";
    };

    auto eb      = 1e-2;
    auto mode    = std::string("abs");
    auto use_r2r = false;

    if (argc < 3) {  //
        help();
    }
    else if (argc >= 3) {
        auto demo = atoi(argv[1]);
        fname     = std::string(argv[2]);
        if (argc >= 4) eb = atof(argv[3]);
        if (argc == 5) mode = std::string(argv[4]);
        use_r2r = mode == "r2r";

        if (demo == 1)
            predictor_demo(false, eb, false, use_r2r);
        else if (demo == 2)
            predictor_demo(true, eb, false, use_r2r);
        else if (demo == 3)
            predictor_demo(true, eb, true, use_r2r);
        else
            help();
    }

    return 0;
}
