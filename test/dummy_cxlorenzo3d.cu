
#include "../src/kernel/lorenzo.cuh"
#include "../src/kernel/prototype_lorenzo.cuh"
#include "../src/metadata.hh"
#include "../src/utils/cuda_err.cuh"

using Data  = float;
using Quant = unsigned short;

__global__ void dummy() { float data = threadIdx.x; }

int main()
{
    LorenzoZipContext zip_ctx;
    zip_ctx.d0      = 512;
    zip_ctx.d1      = 512;
    zip_ctx.d2      = 512;
    zip_ctx.stride1 = 512;
    zip_ctx.stride2 = 512 * 512;
    zip_ctx.radius  = 0;
    zip_ctx.ebx2_r  = 1;

    LorenzoUnzipContext unzip_ctx;
    unzip_ctx.d0      = 512;
    unzip_ctx.d1      = 512;
    unzip_ctx.d2      = 512;
    unzip_ctx.stride1 = 512;
    unzip_ctx.stride2 = 512 * 512;
    unzip_ctx.radius  = 0;
    unzip_ctx.ebx2    = 1;

    auto length = unzip_ctx.d0 * unzip_ctx.d1 * unzip_ctx.d2;

    Data*  data;
    Quant* quant;

    cudaMallocManaged(&data, length * sizeof(Data));
    cudaMallocManaged(&quant, length * sizeof(Quant));

    Data* outlier = data;

    dummy<<<512, 512>>>();
    HANDLE_ERROR(cudaDeviceSynchronize());

    for (auto i = 0; i < 100; i++) {
        // prototype_kernel::c_lorenzo_3d1l<Data, Quant>
        //     <<<dim3(512 / 8, 512 / 8, 512 / 8), dim3(8, 8, 8), 8 * 8 * 8 * sizeof(float)>>>(zip_ctx, data, quant);
        // HANDLE_ERROR(cudaDeviceSynchronize());

        // kernel::c_lorenzo_3d1l_v1_32x8x8data_mapto_32x1x8<Data, Quant>
        //     <<<dim3(512 / 32, 512 / 8, 512 / 8), dim3(32, 1, 8)>>>(zip_ctx, data, quant);
        // HANDLE_ERROR(cudaDeviceSynchronize());

        // legacy_kernel::x_lorenzo_3d1l_v3_8x8x8data_mapto_8x1x8<Data, Quant>
        //     <<<dim3(512 / 8, 512 / 8, 512 / 8), dim3(8, 1, 8)>>>(unzip_ctx, data, outlier, quant);
        // HANDLE_ERROR(cudaDeviceSynchronize());

        kernel::x_lorenzo_3d1l_v4_8x8x8data_mapto_8x1x8<Data, Quant>
            <<<dim3(512 / 8, 512 / 8, 512 / 8), dim3(8, 1, 8)>>>(unzip_ctx, data, outlier, quant);
        HANDLE_ERROR(cudaDeviceSynchronize());

        kernel::x_lorenzo_3d1l_v5_32x8x8data_mapto_32x1x8<Data, Quant>
            <<<dim3(512 / 32, 512 / 8, 512 / 8), dim3(32, 1, 8)>>>(unzip_ctx, data, outlier, quant);
        HANDLE_ERROR(cudaDeviceSynchronize());

        kernel::x_lorenzo_3d1l_v6_32x8x8data_mapto_32x1x8<Data, Quant>
            <<<dim3(512 / 32, 512 / 8, 512 / 8), dim3(32, 1, 8)>>>(unzip_ctx, data, outlier, quant);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    cudaFree(data);
    cudaFree(quant);

    cudaDeviceReset();

    return 0;
}
