/**
 * @file ex_api_pqlorenzo.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-01-08
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "ex_common.cuh"

#include "wrapper/extrap_lorenzo.cuh"

using DATA    = float;
using ERRCTRL = uint16_t;

dim3 const   xyz    = dim3(500, 500, 100);
double const eb     = 1e-4;
int const    radius = 512;

using COMPONENT = cusz::PredictorLorenzo<DATA, ERRCTRL, float>;

void compress_time__alloc_inside(
    COMPONENT&   component,
    DATA*&       origin,
    DATA*&       anchor,
    ERRCTRL*&    errctrl,
    cudaStream_t stream)
{
    auto in_data__out_outlier = origin;

    component.allocate_workspace();
    component.construct(in_data__out_outlier, eb, radius, anchor, errctrl, stream);
}

void decompress_time(COMPONENT& component, DATA*& anchor, ERRCTRL*& errctrl, DATA*& reconstructed, cudaStream_t stream)
{
    auto in_outlier__out_reconsructed = reconstructed;
    component.reconstruct(anchor, errctrl, eb, radius, in_outlier__out_reconsructed, stream);
}

int main(int argc, char** argv)
{
    COMPONENT component(xyz);

    DATA*  experimented_data{nullptr};
    DATA*& reconstructed = experimented_data;
    DATA*  original{nullptr};

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    DATA*    anchor{nullptr};
    ERRCTRL* errctrl{nullptr};

    auto len = xyz.x * xyz.y * xyz.z;

    // ----------------------------------------------------------------------------------------
    exp__prepare_data(&experimented_data, &original, len, 0, 2, false);

    for (auto i = 0; i < len; i++) experimented_data[i] /= 666.7;

    cudaMemcpy(original, experimented_data, sizeof(DATA) * len, cudaMemcpyDeviceToDevice);

    // for (auto i = 0; i < 100; i++) printf("%d\toriginal: %f\treconstructed: %f\n", i, original[i], reconstructed[i]);

    compress_time__alloc_inside(component, experimented_data, anchor, errctrl, stream);
    // ----------------------------------------------------------------------------------------
    decompress_time(component, anchor, errctrl, reconstructed, stream);

    CHECK_CUDA(cudaStreamSynchronize(stream));

    // for (auto i = 0; i < 100; i++) printf("%d\toriginal: %f\treconstructed: %f\n", i, original[i], reconstructed[i]);

    verify_errorboundness(reconstructed, original, eb, len);
    // ----------------------------------------------------------------------------------------
    exp__free(experimented_data, original);
    if (stream) CHECK_CUDA(cudaStreamDestroy(stream));

    return 0;
}
