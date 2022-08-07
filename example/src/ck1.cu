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

#include "cli/quality_viewer.hh"
// #include "cli/timerecord_viewer.hh"
// #include "cusz.h"
#include "kernel/cpplaunch_cuda.hh"

template <typename T, typename E>
void f(std::string fname, double error_bound = 1.2e-4)
{
    /* For demo, we use 3600x1800 CESM data. */
    auto len = 3600 * 1800;

    T *d_d, *h_d;
    T *d_xd, *h_xd;
    T* d_anchor = nullptr;
    E *d_eq, *h_eq;

    /* code snippet for looking at the device array easily */
    auto peek_devdata_T = [](T* d_arr, size_t num = 20) {
        thrust::for_each(thrust::device, d_arr, d_arr + num, [=] __device__ __host__(const T i) { printf("%f\t", i); });
        printf("\n");
    };

    auto peek_devdata_E = [](E* d_arr, size_t num = 20) {
        thrust::for_each(thrust::device, d_arr, d_arr + num, [=] __device__ __host__(const E i) { printf("%u\t", i); });
        printf("\n");
    };

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
    peek_devdata_T(d_d, 20);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    dim3 len3 = dim3(3600, 1800, 1);

    float time;
    cusz::cpplaunch_construct_LorenzoI<T, E, float>(  //
        false, d_d, len3, d_anchor, len3, d_eq, len3, error_bound, 128, &time, stream);

    cudaMemcpy(d_xd, d_d, sizeof(T) * len, cudaMemcpyDeviceToDevice);

    cusz::cpplaunch_reconstruct_LorenzoI<T, E, float>(  //
        d_xd, len3, d_anchor, len3, d_eq, len3, error_bound, 128, &time, stream);

    /* demo: offline checking (de)compression quality. */
    /* load data again    */ cudaMemcpy(d_d, h_d, sizeof(T) * len, cudaMemcpyHostToDevice);
    /* perform evaluation */ cusz::QualityViewer::echo_metric_gpu(d_xd, d_d, len);

    cudaStreamDestroy(stream);

    /* a casual peek */
    printf("peeking xdata, 20 elements\n");
    peek_devdata_T(d_xd, 20);
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        printf("PROG /path/to/cesm-3600x1800 ErrorBound [optional: ErrorQuantType]\n");
        exit(0);
    }
    else {
        auto eb = atof(argv[2]);
        if (string(argv[3]) == "ui8")
            f<float, uint8_t>(std::string(argv[1]), eb);
        else if (string(argv[3]) == "ui16")
            f<float, uint16_t>(std::string(argv[1]), eb);
        else if (string(argv[3]) == "ui32")
            f<float, uint32_t>(std::string(argv[1]), eb);
        else if (string(argv[3]) == "fp32")
            f<float, float>(std::string(argv[1]), eb);
        else
            f<float, uint16_t>(std::string(argv[1]), eb);
    }

    return 0;
}
