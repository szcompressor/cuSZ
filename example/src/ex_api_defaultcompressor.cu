/**
 * @file ex_api_defaultcompressor.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-01-23
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "default_path.cuh"

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "context.hh"
#include "ex_common.cuh"

#include "default_path.cuh"
#include "wrapper/csr11.cuh"
#include "wrapper/extrap_lorenzo.cuh"

#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

using std::cout;

using T = float;
using E = uint16_t;
using H = uint32_t;

constexpr int radius = 512;

int          pardeg = 8192;
unsigned int dimx = 1, dimy = 1, dimz = 1;
dim3         xyz;
unsigned int len = dimx * dimy * dimz;

using Compressor = DefaultPath::DefaultCompressor;
using Predictor  = cusz::PredictorLorenzo<T, E, float>;
using SpReducer  = cusz::CSR11<T>;
using Codec      = cusz::HuffmanCoarse<E, H, uint32_t>;

std::string fname("");

void predictor_detail(T* data, T* cmp, dim3 xyz, double eb, int pardeg, cudaStream_t stream = nullptr)
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

    Predictor predictor(xyz);
    SpReducer spreducer;
    Codec     codec;

    T* xdata = data;

    T* anchor{nullptr};
    E* errctrl{nullptr};

    auto m = Reinterpret1DTo2D::get_square_size(len);

    auto _1_allocate_workspace = [&]() {  //
        printf("_1_allocate_workspace\n");
        predictor.allocate_workspace();
    };

    auto _1_compress_time = [&]() {
        printf("_1_compress_time\n");
        predictor.construct(data, eb, radius, anchor, errctrl, stream), BARRIER();
    };

    auto _1_decompress_time = [&]() {  //
        printf("_1_decompress_time\n");
        predictor.reconstruct(anchor, errctrl, eb, radius, xdata, stream), BARRIER();
    };

    _1_allocate_workspace();
    _1_compress_time();
    _1_decompress_time();

    echo_metric_gpu(xdata, cmp, len);
}

void compressor_detail(T* data, T* cmp, dim3 xyz, double eb, int pardeg, cudaStream_t stream = nullptr)
{
    Compressor compressor(xyz);
    BYTE*      compressed;
    size_t     compressed_len;

    auto xdata = data;

    compressor.allocate_workspace(1024, pardeg, true);
    compressor.compress(data, eb, radius, pardeg, compressed, compressed_len, stream, true /*dbg*/);

    Capsule<BYTE> file(compressed_len);
    file.template alloc<cusz::LOC::HOST_DEVICE>();
    cudaMemcpy(file.dptr, compressed, compressed_len, cudaMemcpyDeviceToDevice);
    cudaMemset(compressed, 0x0, compressed_len);

    cudaMemcpy(compressed, file.dptr, compressed_len, cudaMemcpyDeviceToDevice);
    compressor.decompress(compressed, eb, radius, xdata, stream);

    echo_metric_gpu(xdata, cmp, len);
}

void demo(double eb = 1e-2, bool use_compressor = false, bool use_r2r = false, bool sync_by_stream = false)
{
    Capsule<T> exp(len, "exp data");
    Capsule<T> bak(len, "bak data");

    cout << fname << '\n';

    exp.template alloc<cusz::LOC::HOST_DEVICE, cusz::ALIGNDATA::SQUARE_MATRIX>()
        .template from_file<cusz::LOC::HOST>(fname)
        .host2device();
    bak.template alloc<cusz::LOC::HOST_DEVICE>();
    cudaMemcpy(bak.hptr, exp.hptr, len * sizeof(T), cudaMemcpyHostToHost);
    bak.host2device();

    double adjusted_eb;
    figure_out_eb(exp, eb, adjusted_eb, use_r2r);

    cudaStream_t stream;
    if (sync_by_stream) CHECK_CUDA(cudaStreamCreate(&stream));

    if (stream)
        printf("stream not null\n");
    else
        printf("stream is NULL\n");

    if (not use_compressor) {  //
        predictor_detail(exp.dptr, bak.dptr, xyz, adjusted_eb, pardeg, stream);
    }
    else {
        compressor_detail(exp.dptr, bak.dptr, xyz, adjusted_eb, pardeg, stream);
    }
    if (sync_by_stream and stream) CHECK_CUDA(cudaStreamDestroy(stream));

    exp.template free<cusz::LOC::HOST_DEVICE>();
    bak.template free<cusz::LOC::HOST_DEVICE>();
}

int main(int argc, char** argv)
{
    cudaDeviceReset();

    auto ctx = new cuszCTX(argc, argv);

    auto eb   = (*ctx).eb;
    auto mode = (*ctx).mode;
    auto sync = std::string("stream");
    fname     = (*ctx).fname.fname;

    len  = (*ctx).data_len;
    dimx = (*ctx).x;
    dimy = (*ctx).y;
    dimz = (*ctx).z;
    xyz  = dim3(dimx, dimy, dimz);

    int& chunksize = (*ctx).huffman_chunksize;
    int& pardeg    = (*ctx).nchunk;
    DefaultPath::DefaultBinding::CODEC::get_coarse_pardeg(len, chunksize, pardeg);

    printf("%-*s: %d\n", 16, "pardeg", pardeg);
    printf("%-*s: %d\n", 16, "chunksize", chunksize);

    demo(eb, true, mode == "r2r", sync == "stream");

    return 0;
}
