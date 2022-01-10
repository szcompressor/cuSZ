/**
 * @file ex_api_spline3.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-06-06
 * (create) 2021-06-06 (rev) 2022-01-08
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "ex_common.cuh"

#include "wrapper/interp_spline3.cuh"

#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

using std::cout;

using T = float;
using E = float;

bool print_fullhist = false;
bool write_quant    = false;

double const eb          = 3e-3;
int const    fake_radius = 0;

constexpr unsigned int dimz = 449, dimy = 449, dimx = 235;
constexpr dim3         xyz = dim3(449, 449, 235);
constexpr unsigned int len = dimx * dimy * dimz;

std::string fname;

using Predictor = cusz::Spline3<T, E, float>;

void predictor_demo_detail(T*& original_data, dim3 xyz, cudaStream_t stream)
{
    Predictor predictor;

    T*& reconstructed = original_data;
    T*  anchor{nullptr};
    E*  errctrl{nullptr};

    auto compress_time__alloc_inside = [&]() {
        predictor.allocate_workspace(xyz);
        predictor.construct(original_data, eb, fake_radius, anchor, errctrl, stream);
    };

    auto decompress_time = [&]() {  //
        predictor.reconstruct(anchor, errctrl, eb, fake_radius, reconstructed, stream);
    };

    // -----------------------------------------------------------------------------
    compress_time__alloc_inside();
    decompress_time();
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // for (auto i = 0; i < 100; i++) printf("%d\toriginal: %f\treconstructed: %f\n", i, original_data[i],
    // reconstructed[i]);

    verify_errorboundness(reconstructed, original_data, eb, len);
}

void predictor_demo()
{
    T* experimented_data{nullptr};
    T* original_data{nullptr};

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    exp__prepare_data(&experimented_data, &original_data, len, 0, 2, false);
    for (auto i = 0; i < len; i++) experimented_data[i] /= 666.7;
    cudaMemcpy(original_data, experimented_data, sizeof(T) * len, cudaMemcpyDeviceToDevice);

    predictor_demo_detail(experimented_data, xyz, stream);

    exp__free(experimented_data, original_data);
    if (stream) CHECK_CUDA(cudaStreamDestroy(stream));
}

int main(int argc, char** argv)
{
    predictor_demo();

    return 0;
}
