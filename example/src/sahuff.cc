/**
 * @file sahuff.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-15
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <string>
#include "cli/quality_viewer.hh"
#include "component/codec.hh"
#include "kernel/hist.cuh"
#include "utils/io.hh"

template <typename T, typename H = uint32_t>
void f(std::string fname, size_t const x, size_t const y, size_t const z)
{
    /* For demo, we use 3600x1800 CESM data. */
    auto len = x * y * z;

    T *            d_d, *h_d;
    T *            d_xd, *h_xd;
    uint32_t*      d_freq;
    uint8_t*       d_compressed;
    constexpr auto booklen = 1024;
    constexpr auto pardeg  = 768;
    auto           sublen  = (len - 1) / pardeg + 1;

    /* code snippet for looking at the device array easily */
    auto peek_devdata_T = [](T* d_arr, size_t num = 20) {
        thrust::for_each(
            thrust::device, d_arr, d_arr + num, [=] __device__ __host__(const T i) { printf("%u\t", (uint32_t)i); });
        printf("\n");
    };

    hipMalloc(&d_d, sizeof(T) * len);
    hipMalloc(&d_xd, sizeof(T) * len);
    hipMalloc(&d_freq, sizeof(uint32_t) * booklen);
    hipHostMalloc(&h_d, sizeof(T) * len);
    hipHostMalloc(&h_xd, sizeof(T) * len);

    /* User handles loading from filesystem & transferring to device. */
    io::read_binary_to_array(fname, h_d, len);
    hipMemcpy(d_d, h_d, sizeof(T) * len, hipMemcpyHostToDevice);

    /* a casual peek */
    printf("peeking data, 20 elements\n");
    peek_devdata_T(d_d, 20);

    hipStream_t stream;
    hipStreamCreate(&stream);

    dim3 len3 = dim3(x, y, z);

    float time_hist;

    launch_histogram<T>(d_d, len, d_freq, booklen, time_hist, stream);

    cusz::LosslessCodec<T, H, uint32_t> encoder;
    encoder.init(len, booklen, pardeg /* not optimal for perf */);

    hipMalloc(&d_compressed, len * sizeof(T) / 2);

    // float  time;
    size_t outlen;
    encoder.build_codebook(d_freq, booklen, stream);
    encoder.encode(d_d, len, d_freq, booklen, sublen, pardeg, d_compressed, outlen, stream);

    printf("Huffman in  len:\t%u\n", len);
    printf("Huffman out len:\t%u\n", outlen);
    printf(
        "\"Huffman CR = sizeof(T) * len / outlen\", where outlen is byte count:\t%.2lf\n",
        len * sizeof(T) * 1.0 / outlen);

    encoder.decode(d_compressed, d_xd);

    // hipMemcpy(h_xd, d_xd, len * sizeof(T), hipMemcpyDeviceToHost);
    // /* perform evaluation */ cusz::QualityViewer::identical(CUSZ_CPU, h_xd, h_d, len);
    /* perform evaluation */ cusz::QualityViewer::identical(CUSZ_GPU, d_xd, d_d, len);

    hipStreamDestroy(stream);

    /* a casual peek */
    printf("peeking xdata, 20 elements\n");
    peek_devdata_T(d_xd, 20);
}

int main(int argc, char** argv)
{
    if (argc < 6) {
        printf("PROG /path/to/datafield X Y Z [optional: ErrorQuantType]\n");
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
