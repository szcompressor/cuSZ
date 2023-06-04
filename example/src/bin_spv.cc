/**
 * @file bin_spv.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-21
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

// #include <thrust/device_ptr.h>
// #include <thrust/extrema.h>
// #include <thrust/transform_reduce.h>
// #include <thrust/tuple.h>
#include <string>
#include "kernel/spv_gpu.hh"
#include "pipeline/spcodec.inl"
#include "stat/compare_gpu.hh"
#include "utils/io.hh"
#include "utils/print_gpu.hh"
#include "utils/quality_viewer.hh"

template <typename T, typename H = uint32_t>
void f(std::string fname, size_t const x, size_t const y, size_t const z)
{
    /* For demo, we use 3600x1800 CESM data. */
    auto len = x * y * z;

    T *      d_d, *h_d;
    T *      d_xd, *h_xd;
    uint8_t* d_compressed;

    cudaMalloc(&d_d, sizeof(T) * len);
    cudaMalloc(&d_xd, sizeof(T) * len);
    cudaMallocHost(&h_d, sizeof(T) * len);
    cudaMallocHost(&h_xd, sizeof(T) * len);

    /* User handles loading from filesystem & transferring to device. */
    io::read_binary_to_array(fname, h_d, len);
    cudaMemcpy(d_d, h_d, sizeof(T) * len, cudaMemcpyHostToDevice);

    /* a casual peek */
    printf("peeking data, 20 elements\n");
    psz::peek_device_data<T>(d_d, 20);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    dim3 len3 = dim3(x, y, z);

    cusz::SpcodecVec<T, uint32_t> encoder;
    encoder.init(len, 1);  // CR lower bound = 1

    cudaMalloc(&d_compressed, len * sizeof(T) * 2);

    // float  time;
    size_t outlen;
    encoder.encode(d_d, len, d_compressed, outlen, stream, true);

    printf("Spvec in  len:\t%u\n", len);
    printf("Spvec out len:\t%u\n", outlen);
    printf(
        "\"Spvec CR = sizeof(T) * len / outlen\", where outlen is byte count:\t%.2lf\n",
        len * sizeof(T) * 1.0 / outlen);

    encoder.decode(d_compressed, d_xd, stream);

    /* perform evaluation */ auto identical = psz::thrustgpu_identical(d_xd, d_d, len);

    if (identical)
        cout << ">>>>  IDENTICAL." << endl;
    else
        cout << "!!!!  ERROR: NOT IDENTICAL." << endl;

    cudaStreamDestroy(stream);

    /* a casual peek */
    printf("peeking xdata, 20 elements\n");
    psz::peek_device_data<T>(d_xd, 20);
}

int main(int argc, char** argv)
{
    if (argc < 6) {
        printf("PROG /path/to/datafield X Y Z [InputType]\n");
        printf("0    1                  2 3 4 5\n");
        exit(0);
    }
    else {
        auto fname = std::string(argv[1]);
        auto x     = atoi(argv[2]);
        auto y     = atoi(argv[3]);
        auto z     = atoi(argv[4]);
        auto type  = std::string(argv[5]);

        if (type == "ui8")
            f<uint8_t, uint32_t>(fname, x, y, z);
        else if (type == "ui16")
            f<uint16_t, uint32_t>(fname, x, y, z);
        else if (type == "ui32")
            f<uint32_t, uint32_t>(fname, x, y, z);
        else
            f<uint16_t, uint32_t>(fname, x, y, z);
    }

    return 0;
}