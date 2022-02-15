/**
 * @file app_spline3.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2022-02-12
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

constexpr int FAKE_RADIUS = 0;
constexpr int radius      = FAKE_RADIUS;

unsigned int dimx = 1, dimy = 1, dimz = 1;
dim3         xyz() { return dim3(dimx, dimy, dimz); }
unsigned int len() { return dimx * dimy * dimz; }

using Compressor = SparsityAwarePath::DefaultCompressor;
using Predictor  = cusz::Spline3<T, E, float>;
using SpReducer  = cusz::CSR11<T>;

int         quant_radius = 512;
bool        gpu_verify   = true;
std::string fname("");

void predictor_detail(T* data, T* cmp, dim3 _xyz, double eb, cudaStream_t stream = nullptr, int real_radius = 512)
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

    Predictor predictor(_xyz, true);
    SpReducer spreducer;

    T* xdata = data;

    T* anchor{nullptr};
    E* errctrl{nullptr};

    // extra space for outputing
    Capsule<int32_t> uint_quant(len());
    uint_quant.template alloc<cusz::LOC::HOST_DEVICE>();  // implicit memset

    auto dbg_echo_nnz = [&]() {
        int __nnz = thrust::count_if(
            thrust::device, errctrl, errctrl + predictor.get_quant_footprint(),
            [] __device__(const T& x) { return x != 0; });
        cout << "__nnz: " << __nnz << '\n';
    };

    auto export_quant_code = [&]() {
        thrust::transform_if(
            thrust::device,                                                               //
            errctrl, errctrl + len(),                                                     //
            uint_quant.dptr,                                                              //
            [=] __device__(const E e) { return static_cast<int32_t>(e + real_radius); },  //
            [=] __device__(const E e) { return abs(e) < real_radius; }                    //
        );

        // thrust::for_each(
        //     thrust::device, errctrl, errctrl + len(),  //
        //     [=] __device__ __host__(const E e) {
        //         if (e == 6 or e == 5) printf("%.0f\t", e);
        //     });
        // cout << '\n';
        // cout << '\n';

        // thrust::for_each(
        //     thrust::device, uint_quant.dptr, uint_quant.dptr + len(),  //
        //     [=] __device__ __host__(const int32_t q) {
        //         if (q == 6 + real_radius or q == 5 + real_radius) printf("%d\t", q - real_radius);
        //     });
        // cout << '\n';
        // cout << '\n';
    };

    auto _1_allocate_workspace = [&]() {  //
        printf("_1_allocate_workspace\n");
        predictor.allocate_workspace();

    };

    auto _1_compress_time = [&]() {
        printf("_1_compress_time\n");
        predictor.construct(data, eb, radius, anchor, errctrl, stream);
        BARRIER();

        export_quant_code();
        BARRIER();

        cout << "export integer quant code:\t" << fname + ".quant\n";
        uint_quant.device2host().to_file<cusz::LOC::HOST>(fname + ".quant");

        dbg_echo_nnz();
    };

    auto _1_decompress_time = [&]() {  //
        printf("_1_decompress_time\n");
        predictor.reconstruct(anchor, errctrl, eb, radius, xdata, stream);
        BARRIER();
    };

    // -----------------------------------------------------------------------------

    _1_allocate_workspace();
    _1_compress_time();
    _1_decompress_time();
    uint_quant.template free<cusz::LOC::HOST_DEVICE>();

    if (gpu_verify)
        echo_metric_gpu(xdata, cmp, len());
    else
        echo_metric_cpu(xdata, cmp, len(), true);
}

void predictor_demo(double eb = 1e-2, bool use_r2r = false, int real_radius = 512)
{
    if (len() == 1) { throw std::runtime_error("Length (in 1D) must not be 1."); }

    Capsule<T> exp(len(), "exp data");
    Capsule<T> bak(len(), "bak data");

    exp.template alloc<cusz::LOC::HOST_DEVICE>().template from_file<cusz::LOC::HOST>(fname).host2device();
    bak.template alloc<cusz::LOC::HOST_DEVICE>();
    cudaMemcpy(bak.hptr, exp.hptr, len() * sizeof(T), cudaMemcpyHostToHost);
    bak.host2device();

    double adjusted_eb;
    figure_out_eb(exp, eb, adjusted_eb, use_r2r);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    predictor_detail(exp.dptr, bak.dptr, xyz(), adjusted_eb, stream, real_radius);
    if (stream) CHECK_CUDA(cudaStreamDestroy(stream));

    exp.template free<cusz::LOC::HOST_DEVICE>();
    bak.template free<cusz::LOC::HOST_DEVICE>();
}

int main(int argc, char** argv)
{
    auto help = []() {
        printf("./prog <1-3:x,y,z> <4:fname> [5:eb = 1e-2] [6:mode = abs] [7:verify = gpu]\n");
        printf("<..> necessary, [..] optional\n");
        printf(
            "argv[1]: x\n"
            "argv[2]: y\n"
            "argv[3]: z\n"
            "argv[4]: filename\n"
            "argv[5]: error bound (default to \"1e-2\")\n"
            "argv[6]: mode, abs or r2r (default to \"abs\")\n"
            "argv[7]: if using GPU to verify (default to \"gpu\")\n"
            "argv[8]: real radius for seperating (uint32) quant-code from (fp) errctrl-code (default to \"512\")\n");
    };

    double eb      = 1e-2;
    bool   use_r2r = false;

    if (argc < 5) {  //
        help();
    }
    else if (argc >= 5) {
        /* required */ dimx  = atoi(argv[1]);
        /* required */ dimy  = atoi(argv[2]);
        /* required */ dimz  = atoi(argv[3]);
        /* required */ fname = std::string(argv[4]);

        /* optional */
        if (argc >= 6) eb = atof(argv[5]);
        if (argc >= 7) use_r2r = std::string(argv[6]) == "r2r";
        if (argc >= 8) gpu_verify = std::string(argv[7]) == "gpu";
        if (argc == 9) quant_radius = atoi(argv[8]);

        cout << "fname:\t" << fname << '\n';
        cout << "eb:\t" << eb << "\tmode:\t" << argv[6] << '\n';

        predictor_demo(eb, use_r2r, quant_radius);
    }

    return 0;
}
