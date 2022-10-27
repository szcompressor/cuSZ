/**
 * @file ck2.cu
 * @author Boyuan Zhang, Jiannan Tian
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

#include "nvcomp.hpp"
#include "nvcomp/lz4.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

using namespace nvcomp;

#define CUDA_CHECK(cond)                         \
    do {                                         \
        cudaError_t err = cond;                  \
        if (err != cudaSuccess) {                \
            std::cerr << "Failure" << std::endl; \
            exit(1);                             \
        }                                        \
    } while (false)

void comp_decomp_with_single_manager(
    uint8_t*     device_input_ptrs,
    const size_t input_buffer_len,
    uint8_t*     res_decomp_buffer,
    cudaStream_t stream)
{
    // cudaStream_t stream;
    // CUDA_CHECK(cudaStreamCreate(&stream));

    const int    chunk_size = 1 << 16;
    nvcompType_t data_type  = NVCOMP_TYPE_CHAR;

    LZ4Manager        nvcomp_manager{chunk_size, data_type, stream};
    CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);

    uint8_t* comp_buffer;
    CUDA_CHECK(cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));

    nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);
    size_t compressedSize = nvcomp_manager.get_compressed_output_size(comp_buffer);
    printf("compression ratio (origin size / compressed size): %f\n", float(input_buffer_len) / float(compressedSize));

    DecompressionConfig decomp_config = nvcomp_manager.configure_decompression(comp_buffer);
    // uint8_t* res_decomp_buffer;
    // CUDA_CHECK(cudaMalloc(&res_decomp_buffer, decomp_config.decomp_data_size));

    nvcomp_manager.decompress(res_decomp_buffer, comp_buffer, decomp_config);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFree(comp_buffer));
    // CUDA_CHECK(cudaFree(res_decomp_buffer));

    // CUDA_CHECK(cudaStreamDestroy(stream));
}

void comp_decomp_with_single_manager_with_checksums(
    uint8_t*     device_input_ptrs,
    const size_t input_buffer_len,
    uint8_t*     res_decomp_buffer,
    cudaStream_t stream)
{
    //   cudaStream_t stream;
    //   CUDA_CHECK(cudaStreamCreate(&stream));

    const int    chunk_size = 1 << 16;
    nvcompType_t data_type  = NVCOMP_TYPE_CHAR;

    /*
     * There are 5 possible modes for checksum processing as
     * described below.
     *
     * Mode: NoComputeNoVerify
     * Description:
     *   - During compression, do not compute checksums
     *   - During decompression, do not verify checksums
     *
     * Mode: ComputeAndNoVerify
     * Description:
     *   - During compression, compute checksums
     *   - During decompression, do not attempt to verify checksums
     *
     * Mode: NoComputeAndVerifyIfPresent
     * Description:
     *   - During compression, do not compute checksums
     *   - During decompression, verify checksums if they were included
     *
     * Mode: ComputeAndVerifyIfPresent
     * Description:
     *   - During compression, compute checksums
     *   - During decompression, verify checksums if they were included
     *
     * Mode: ComputeAndVerify
     * Description:
     *   - During compression, compute checksums
     *   - During decompression, verify checksums. A runtime error will be
     *     thrown upon configure_decompression if checksums were not
     *     included in the compressed buffer.
     */

    int gpu_num = 0;

    // manager constructed with checksum mode as final argument
    LZ4Manager        nvcomp_manager{chunk_size, data_type, stream, gpu_num, ComputeAndVerify};
    CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);

    uint8_t* comp_buffer;
    CUDA_CHECK(cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));

    // Checksums are computed and stored for uncompressed and compressed buffers during compression
    nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);

    DecompressionConfig decomp_config = nvcomp_manager.configure_decompression(comp_buffer);
    //   uint8_t* res_decomp_buffer;
    //   CUDA_CHECK(cudaMalloc(&res_decomp_buffer, decomp_config.decomp_data_size));

    // Checksums are computed for compressed and decompressed buffers and verified against those
    // stored during compression
    nvcomp_manager.decompress(res_decomp_buffer, comp_buffer, decomp_config);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     * After synchronizing the stream, the nvcomp status can be checked to see if
     * the checksums were successfully verified. Provided no unrelated nvcomp errors occurred,
     * if the checksums were successfully verified, the status will be nvcompSuccess. Otherwise,
     * it will be nvcompErrorBadChecksum.
     */
    nvcompStatus_t final_status = *decomp_config.get_status();
    if (final_status == nvcompErrorBadChecksum) { throw std::runtime_error("One or more checksums were incorrect.\n"); }

    CUDA_CHECK(cudaFree(comp_buffer));
    //   CUDA_CHECK(cudaFree(res_decomp_buffer));

    //   CUDA_CHECK(cudaStreamDestroy(stream));
}

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

    T *       d_d, *h_d;
    T *       d_xd, *h_xd;
    T*        d_outlier;
    E *       d_eq, *h_eq;
    T*        d_anchor    = nullptr;
    uint32_t* outlier_idx = nullptr;
    dim3      len3        = dim3(x, y, z);
    dim3      dummy_len3  = dim3(0, 0, 0);

    cudaMalloc(&d_d, sizeof(T) * len);
    cudaMalloc(&d_outlier, sizeof(T) * len);
    cudaMalloc(&d_xd, sizeof(T) * len);
    cudaMalloc(&d_eq, sizeof(E) * len);
    cudaMallocHost(&h_d, sizeof(T) * len);
    cudaMallocHost(&h_xd, sizeof(T) * len);
    cudaMallocHost(&h_eq, sizeof(E) * len);

    /* User handles loading from filesystem & transferring to device. */
    io::read_binary_to_array(fname, h_d, len);
    cudaMemcpy(d_d, h_d, sizeof(T) * len, cudaMemcpyHostToDevice);

    /* a casual peek */
    peek_device_data<T>(d_d, 100);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float time;

    if (not use_proto) {
        cout << "using optimized comp. kernel\n";
        cusz::cpplaunch_construct_LorenzoI<T, E, FP>(                        //
            d_d, len3, error_bound, radius,                                  // input and config
            d_eq, dummy_len3, d_anchor, dummy_len3, d_outlier, outlier_idx,  // output
            &time, stream);
    }
    else {
        cout << "using prototype comp. kernel\n";
        cusz::cpplaunch_construct_LorenzoI_proto<T, E, FP>(                  //
            d_d, len3, error_bound, radius,                                  // input and config
            d_eq, dummy_len3, d_anchor, dummy_len3, d_outlier, outlier_idx,  // output
            &time, stream);
    }

    cudaDeviceSynchronize();

    peek_device_data<E>(d_eq, 100);

    cudaMemcpy(d_xd, d_d, sizeof(T) * len, cudaMemcpyDeviceToDevice);

    E* d_nvout;
    CUDA_CHECK(cudaMalloc(&d_nvout, sizeof(E) * len));
    comp_decomp_with_single_manager((uint8_t*)d_eq, sizeof(E) * len, (uint8_t*)d_nvout, stream);

    if (not use_proto) {
        cout << "using optimized decomp. kernel\n";
        cusz::cpplaunch_reconstruct_LorenzoI<T, E, FP>(                      //
            d_eq, dummy_len3, d_anchor, dummy_len3, d_outlier, outlier_idx,  // input
            error_bound, radius,                                             // input (config)
            d_xd, len3,                                                      // output
            &time, stream);
    }
    else {
        cout << "using prototype decomp. kernel\n";
        cusz::cpplaunch_reconstruct_LorenzoI_proto<T, E, FP>(                //
            d_eq, dummy_len3, d_anchor, dummy_len3, d_outlier, outlier_idx,  // input
            error_bound, radius,                                             // input (config)
            d_xd, len3,                                                      // output
            &time, stream);
    }

    cudaDeviceSynchronize();

    /* demo: offline checking (de)compression quality. */
    // /* load data again    */ cudaMemcpy(d_d, h_d, sizeof(T) * len, cudaMemcpyHostToDevice);
    /* perform evaluation */ cusz::QualityViewer::echo_metric_gpu(d_xd, d_d, len);

    cudaStreamDestroy(stream);

    /* a casual peek */
    peek_device_data<T>(d_xd, 100);

    cudaFree(d_d);
    cudaFree(d_xd);
    cudaFree(d_eq);
    cudaFree(d_nvout);
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
