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
    auto len = x * y * z;

    T *d_d, *h_d;
    T *d_xd, *h_xd;
    T* d_anchor = nullptr;
    E *d_eq, *h_eq;

    hipMalloc(&d_d, sizeof(T) * len);
    hipMalloc(&d_xd, sizeof(T) * len);
    hipMalloc(&d_eq, sizeof(E) * len);
    hipHostMalloc(&h_d, sizeof(T) * len);
    hipHostMalloc(&h_xd, sizeof(T) * len);
    hipHostMalloc(&h_eq, sizeof(E) * len);

    /* User handles loading from filesystem & transferring to device. */
    io::read_binary_to_array(fname, h_d, len);
    hipMemcpy(d_d, h_d, sizeof(T) * len, hipMemcpyHostToDevice);

    /* a casual peek */
    printf("peeking data, 20 elements\n");
    peek_device_data<T>(d_d, 100);

    hipStream_t stream;
    hipStreamCreate(&stream);

    dim3 len3 = dim3(x, y, z);

    float time;

    if (not use_proto) {
        cout << "using optimized comp. kernel\n";
        cusz::cpplaunch_construct_LorenzoI<T, E, float>(  //
            false, d_d, len3, d_anchor, len3, d_eq, len3, error_bound, radius, &time, stream);
    }
    else {
        cout << "using prototype comp. kernel\n";
        cusz::cpplaunch_construct_LorenzoI_proto<T, E, float>(  //
            false, d_d, len3, d_anchor, len3, d_eq, len3, error_bound, radius, &time, stream);
    }

    cudaDeviceSynchronize();

    peek_device_data<E>(d_eq, 100);

    hipMemcpy(h_eq, d_eq, sizeof(E) * len, hipMemcpyDeviceToHost);
    io::write_array_to_binary<E>(fname + ".eq." + type_literal, h_eq, len);

    hipMemcpy(d_xd, d_d, sizeof(T) * len, hipMemcpyDeviceToDevice);

    if (not use_proto) {
        cout << "using optimized decomp. kernel\n";
        cusz::cpplaunch_reconstruct_LorenzoI<T, E, float>(  //
            d_xd, len3, d_anchor, len3, d_eq, len3, error_bound, radius, &time, stream);
    }
    else {
        cout << "using prototype decomp. kernel\n";
        cusz::cpplaunch_reconstruct_LorenzoI_proto<T, E, float>(  //
            d_xd, len3, d_anchor, len3, d_eq, len3, error_bound, radius, &time, stream);
    }

    cudaDeviceSynchronize();

    /* demo: offline checking (de)compression quality. */
    /* load data again    */ hipMemcpy(d_d, h_d, sizeof(T) * len, hipMemcpyHostToDevice);
    /* perform evaluation */ cusz::QualityViewer::echo_metric_gpu(d_xd, d_d, len);

    hipStreamDestroy(stream);

    /* a casual peek */
    printf("peeking xdata, 20 elements\n");
    peek_device_data<T>(d_xd, 100);
}

int main(int argc, char** argv)
{
    if (argc < 6) {
        printf("PROG /path/to/datafield X Y Z ErrorBound [ErrorQuantType] [Radius] [Use Prototype]\n");
        printf("0    1                  2 3 4 5          [6:ui16]         [7:128]  [8:yes]\n");
        exit(0);
    }
    else {
        auto fname = std::string(argv[1]);
        auto x     = atoi(argv[2]);
        auto y     = atoi(argv[3]);
        auto z     = atoi(argv[4]);
        auto eb    = atof(argv[5]);

        std::string type;
        if (argc > 6)
            type = std::string(argv[6]);
        else
            type = "ui16";
        type_literal = type;

        int radius;
        if (argc > 7)
            radius = atoi(argv[7]);
        else
            radius = 128;

        bool use_prototype;
        if (argc > 8)
            use_prototype = std::string(argv[8]) == "yes";
        else
            use_prototype = false;

        auto radius_legal = [&](int const sizeof_T) {
            size_t upper_bound = 1lu << (sizeof_T * 8);
            cout << upper_bound << endl;
            cout << radius * 2 << endl;
            if ((radius * 2) > upper_bound) throw std::runtime_error("Radius overflows error-quantization type.");
        };

        if (type == "ui8") {
            radius_legal(1);
            f<float, uint8_t>(fname, x, y, z, eb, radius, use_prototype);
        }
        else if (type == "ui16") {
            radius_legal(2);
            f<float, uint16_t>(fname, x, y, z, eb, radius, use_prototype);
        }
        else if (type == "ui32") {
            radius_legal(4);
            f<float, uint32_t>(fname, x, y, z, eb, radius, use_prototype);
        }
        else if (type == "fp32") {
            f<float, float>(fname, x, y, z, eb, radius, use_prototype);
        }
    }

    return 0;
}
