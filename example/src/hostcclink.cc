#include <cuda_runtime.h>
#include "kernel/lorenzo_all.hh"

int main()
{
    auto len3 = dim3(3600, 1800, 1);
    auto len  = 6480000;

    float*    data;
    uint16_t* errq;
    float*    outlier;
    uint32_t* outlier_idx;  // future use
    auto      eb     = 1e-4;
    auto      radius = 512;
    float     time_elapsed;

    cudaMallocManaged(&data, sizeof(float) * len);
    cudaMallocManaged(&outlier, sizeof(float) * len);
    cudaMallocManaged(&errq, sizeof(uint16_t) * len);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    compress_predict_lorenzo_i<float, uint16_t, float>(
        data, len3, eb, radius, errq, dim3(1, 1, 1), nullptr, dim3(1, 1, 1), outlier, outlier_idx, nullptr,
        &time_elapsed, stream);

    cudaFree(data);
    cudaFree(outlier);
    cudaFree(errq);

    cudaStreamDestroy(stream);

    return 0;
}
