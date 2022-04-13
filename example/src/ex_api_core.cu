/**
 * @file ex_api_core.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-10
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "api.hh"

#include "cli/quality_viewer.hh"
#include "cli/timerecord_viewer.hh"

template <typename T>
void f(std::string fname)
{
    using Compressor = typename cusz::Framework<T>::LorenzoFeaturedCompressor;

    /* For demo, we use 3600x1800 CESM data. */
    auto len = 3600 * 1800;

    Compressor*  compressor;
    cusz::Header header;
    BYTE*        compressed;
    size_t       compressed_len;

    T *d_uncompressed, *h_uncompressed;
    T *d_decompressed, *h_decompressed;

    /* cuSZ requires a 3% overhead on device (not required on host). */
    size_t uncompressed_alloclen = len * 1.03;
    size_t decompressed_alloclen = uncompressed_alloclen;

    auto peek_devdata = [](T* d_arr, size_t num = 20) {
        thrust::for_each(thrust::device, d_arr, d_arr + num, [=] __device__ __host__(const T i) { printf("%f\t", i); });
        printf("\n");
    };

    // clang-format off
    cudaMalloc(     &d_uncompressed, sizeof(T) * uncompressed_alloclen );
    cudaMallocHost( &h_uncompressed, sizeof(T) * len );
    cudaMalloc(     &d_decompressed, sizeof(T) * decompressed_alloclen );
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

    compressor = new Compressor;
    BYTE* exposed_compressed;
    {
        cusz::TimeRecord timerecord;
        cusz::Context*   ctx;

        /*
         * Two methods to build the configuration.
         * Note that specifying type is not needed because of T in cusz::Framework<T>
         */

        /* Method 1: Everthing is in string. */
        // char const* config = "eb=2.4e-4,mode=r2r,len=3600x1800";
        // ctx                = new cusz::Context(config);

        /* Method 2: Numeric and string options are set separatedly. */
        ctx = new cusz::Context();
        ctx->set_len(3600, 1800, 1, 1)        // In this case, the last 2 arguments can be omitted.
            .set_eb(2.4e-4)                   // numeric
            .set_control_string("mode=r2r");  // string

        cusz::core_compress(
            compressor, ctx,                             // compressor & config
            d_uncompressed, uncompressed_alloclen,       // input
            exposed_compressed, compressed_len, header,  // output
            stream, &timerecord);

        /* User can interpret the collected time information in other ways. */
        cusz::TimeRecordViewer::view_compression(&timerecord, len * sizeof(T), compressed_len);

        /* verify header */
        // clang-format off
        printf("header.%-*s : %x\n",            12, "(addr)", &header);
        printf("header.%-*s : %lu, %lu, %lu\n", 12, "{x,y,z}", header.x, header.y, header.z);
        printf("header.%-*s : %lu\n",           12, "filesize", header.get_filesize());
        // clang-format on
    }

    /* If needed, User should perform a memcopy to transfer `exposed_compressed` before `compressor` is destroyed. */
    cudaMalloc(&compressed, compressed_len);
    cudaMemcpy(compressed, exposed_compressed, compressed_len, cudaMemcpyDeviceToDevice);

    /* release compressor */ delete compressor;

    compressor = new Compressor;
    {
        cusz::TimeRecord timerecord;

        cusz::core_decompress(
            compressor, &header,                    // compressor & config
            compressed, compressed_len,             // input
            d_decompressed, decompressed_alloclen,  // output
            stream, &timerecord);

        /* User can interpret the collected time information in other ways. */
        cusz::TimeRecordViewer::view_decompression(&timerecord, len * sizeof(T));
    }

    /* a casual peek */
    printf("peeking decompressed data, 20 elements\n");
    peek_devdata(d_decompressed, 20);

    /* demo: offline checking (de)compression quality. */
    /* load data again    */ cudaMemcpy(d_uncompressed, h_uncompressed, sizeof(T) * len, cudaMemcpyHostToDevice);
    /* perform evaluation */ cusz::QualityViewer::echo_metric_gpu(d_decompressed, d_uncompressed, len, compressed_len);

    cudaFree(compressed);
    delete compressor;

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