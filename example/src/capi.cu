/**
 * @file capi.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-05-06
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "cusz.h"
#include "cuszapi.hh"

#include "cli/quality_viewer.hh"
#include "cli/timerecord_viewer.hh"

template <typename T>
void f(std::string fname)
{
    /* For demo, we use 3600x1800 CESM data. */
    auto len = 3600 * 1800;

    cusz_header header;
    uint8_t*    exposed_compressed;
    uint8_t*    compressed;
    size_t      compressed_len;

    T *d_uncompressed, *h_uncompressed;
    T *d_decompressed, *h_decompressed;

    /* cuSZ requires a 3% overhead on device (not required on host). */
    size_t uncompressed_memlen = len * 1.03;
    size_t decompressed_memlen = uncompressed_memlen;

    /* code snippet for looking at the device array easily */
    auto peek_devdata = [](T* d_arr, size_t num = 20) {
        thrust::for_each(thrust::device, d_arr, d_arr + num, [=] __device__ __host__(const T i) { printf("%f\t", i); });
        printf("\n");
    };

    // clang-format off
    cudaMalloc(     &d_uncompressed, sizeof(T) * uncompressed_memlen );
    cudaMallocHost( &h_uncompressed, sizeof(T) * len );
    cudaMalloc(     &d_decompressed, sizeof(T) * decompressed_memlen );
    cudaMallocHost( &h_decompressed, sizeof(T) * len );
    // clang-format on

    /* User handles loading from filesystem & transferring to device. */
    io::read_binary_to_array(fname, h_uncompressed, len);
    cudaMemcpy(d_uncompressed, h_uncompressed, sizeof(T) * len, cudaMemcpyHostToDevice);

    /* a casual peek */
    printf("peeking uncompressed data, 20 elements\n");
    peek_devdata(d_uncompressed, 20);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cusz_framework*  framework  = cusz_default_framework();
    cusz_compressor* comp       = cusz_create(framework, FP32);
    cusz_config*     config     = new cusz_config{.eb = 2.4e-4, .mode = Rel};
    cusz_len         uncomp_len = cusz_len{3600, 1800, 1, 1, 1.03};
    cusz_len         decomp_len = uncomp_len;

    {
        cusz_compress(
            comp, config, d_uncompressed, uncomp_len, &exposed_compressed, &compressed_len, &header, nullptr, stream);

        /* verify header */
        printf("header.%-*s : %x\n", 12, "(addr)", &header);
        printf("header.%-*s : %lu, %lu, %lu\n", 12, "{x,y,z}", header.x, header.y, header.z);
        printf("header.%-*s : %lu\n", 12, "filesize", ConfigHelper::get_filesize(&header));
    }

    /* If needed, User should perform a memcopy to transfer `exposed_compressed` before `compressor` is destroyed. */
    cudaMalloc(&compressed, compressed_len);
    cudaMemcpy(compressed, exposed_compressed, compressed_len, cudaMemcpyDeviceToDevice);

    {
        cusz_decompress(comp, &header, exposed_compressed, compressed_len, d_decompressed, decomp_len, nullptr, stream);
    }

    /* a casual peek */
    printf("peeking decompressed data, 20 elements\n");
    peek_devdata(d_decompressed, 20);

    /* demo: offline checking (de)compression quality. */
    /* load data again    */ cudaMemcpy(d_uncompressed, h_uncompressed, sizeof(T) * len, cudaMemcpyHostToDevice);
    /* perform evaluation */ cusz::QualityViewer::echo_metric_gpu(d_decompressed, d_uncompressed, len, compressed_len);

    cusz_release(comp);

    cudaFree(compressed);
    // delete compressor;

    cudaStreamDestroy(stream);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("PROG /path/to/cesm-3600x1800\n");
        exit(0);
    }

    f<float>(std::string(argv[1]));
    return 0;
}