/**
 * @file capi.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-06
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "cli/quality_viewer.hh"
// #include "cli/timerecord_viewer.hh"
// #include "cusz.h"
#include "kernel/cpplaunch_cuda.hh"
#include "utils/print.cuh"

std::string type_literal;

template <typename T, typename E>
void f(
    std::string const fname,
    size_t const      x,
    size_t const      y,
    size_t const      z,
    double const      error_bound = 1.2e-4,
    int const         radius      = 128,
    bool              use_proto   = false)
{
    // When the input type is FP<X>, the internal precision should be the same.
    using FP = T;

    auto len = x * y * z;

    T *d_d, *h_d;
    T *d_xd, *h_xd;
    T* d_anchor = nullptr;
    E *d_eq, *h_eq;

    cudaMalloc(&d_d, sizeof(T) * len);
    cudaMalloc(&d_xd, sizeof(T) * len);
    cudaMalloc(&d_eq, sizeof(E) * len);
    cudaMallocHost(&h_d, sizeof(T) * len);
    cudaMallocHost(&h_xd, sizeof(T) * len);
    cudaMallocHost(&h_eq, sizeof(E) * len);

    /* User handles loading from filesystem & transferring to device. */
    io::read_binary_to_array(fname, h_d, len);
    cudaMemcpy(d_d, h_d, sizeof(T) * len, cudaMemcpyHostToDevice);

    /* a casual peek */
    printf("peeking data, 20 elements\n");
    peek_device_data<T>(d_d, 100);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    dim3 len3 = dim3(x, y, z);

    float time;

    if (not use_proto) {
        cout << "using optimized comp. kernel\n";
        cusz::cpplaunch_construct_LorenzoI<T, E, FP>(  //
            d_d, len3, d_anchor, len3, d_eq, len3, d_d, error_bound, radius, &time, stream);
    }
    else {
        cout << "using prototype comp. kernel\n";
        cusz::cpplaunch_construct_LorenzoI_proto<T, E, FP>(  //
            d_d, len3, d_anchor, len3, d_eq, len3, d_d, error_bound, radius, &time, stream);
    }

    cudaDeviceSynchronize();

    peek_device_data<E>(d_eq, 100);

    cudaMemcpy(h_eq, d_eq, sizeof(E) * len, cudaMemcpyDeviceToHost);
    io::write_array_to_binary<E>(fname + ".eq." + type_literal, h_eq, len);

    cudaMemcpy(d_xd, d_d, sizeof(T) * len, cudaMemcpyDeviceToDevice);

    if (not use_proto) {
        cout << "using optimized decomp. kernel\n";
        cusz::cpplaunch_reconstruct_LorenzoI<T, E, FP>(  //
            d_xd, len3, d_anchor, len3, d_eq, len3, d_xd, error_bound, radius, &time, stream);
    }
    else {
        cout << "using prototype decomp. kernel\n";
        cusz::cpplaunch_reconstruct_LorenzoI_proto<T, E, FP>(  //
            d_xd, len3, d_anchor, len3, d_eq, len3, d_xd, error_bound, radius, &time, stream);
    }

    cudaDeviceSynchronize();

    /* demo: offline checking (de)compression quality. */
    /* load data again    */ cudaMemcpy(d_d, h_d, sizeof(T) * len, cudaMemcpyHostToDevice);
    /* perform evaluation */ cusz::QualityViewer::echo_metric_gpu(d_xd, d_d, len);

    cudaStreamDestroy(stream);

    /* a casual peek */
    printf("peeking xdata, 20 elements\n");
    peek_device_data<T>(d_xd, 100);

    cudaFree(d_d);
    cudaFree(d_xd);
    cudaFree(d_eq);
    cudaFreeHost(h_d);
    cudaFreeHost(h_xd);
    cudaFreeHost(h_eq);
}

int main(int argc, char** argv)
{
    //// help
    if (argc < 6) {
        printf("0    1             2     3 4 5 6  [7]      [8:128]  [9:yes]\n");
        printf("PROG /path/to/file DType X Y Z EB [EType]  [Radius] [Use Prototype]\n");
        printf(" 2  DType: \"F\" for `float`, \"D\" for `double`\n");
        printf("[7] EType: \"ui{8,16,32}\" for `uint{8,16,32}_t` as quant-code type\n");
        exit(0);
    }

    //// read argv
    auto fname = std::string(argv[1]);
    auto dtype = std::string(argv[2]);
    auto x     = atoi(argv[3]);
    auto y     = atoi(argv[4]);
    auto z     = atoi(argv[5]);
    auto eb    = atof(argv[6]);

    std::string etype;
    if (argc > 7)
        etype = std::string(argv[7]);
    else
        etype = "ui16";
    type_literal = etype;

    int radius;
    if (argc > 8)
        radius = atoi(argv[8]);
    else
        radius = 128;

    bool use_prototype;
    if (argc > 9)
        use_prototype = std::string(argv[9]) == "yes";
    else
        use_prototype = false;

    //// dispatch

    auto radius_legal = [&](int const sizeof_T) {
        size_t upper_bound = 1lu << (sizeof_T * 8);
        cout << upper_bound << endl;
        cout << radius * 2 << endl;
        if ((radius * 2) > upper_bound) throw std::runtime_error("Radius overflows error-quantization type.");
    };

    if (dtype == "F") {
        if (etype == "ui8") {
            radius_legal(1);
            f<float, uint8_t>(fname, x, y, z, eb, radius, use_prototype);
        }
        else if (etype == "ui16") {
            radius_legal(2);
            f<float, uint16_t>(fname, x, y, z, eb, radius, use_prototype);
        }
        else if (etype == "ui32") {
            radius_legal(4);
            f<float, uint32_t>(fname, x, y, z, eb, radius, use_prototype);
        }
        else if (etype == "fp32") {
            f<float, float>(fname, x, y, z, eb, radius, use_prototype);
        }
    }
    else if (dtype == "D") {
        if (etype == "ui8") {
            radius_legal(1);
            f<double, uint8_t>(fname, x, y, z, eb, radius, use_prototype);
        }
        else if (etype == "ui16") {
            radius_legal(2);
            f<double, uint16_t>(fname, x, y, z, eb, radius, use_prototype);
        }
        else if (etype == "ui32") {
            radius_legal(4);
            f<double, uint32_t>(fname, x, y, z, eb, radius, use_prototype);
        }
        else if (etype == "fp32") {
            f<double, float>(fname, x, y, z, eb, radius, use_prototype);
        }
    }
    else
        throw std::runtime_error("not a valid dtype.");

    return 0;
}
