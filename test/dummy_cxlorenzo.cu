
#include <iostream>
#include "../src/kernel/lorenzo.h"
#include "../src/kernel/prototype_lorenzo.cuh"
#include "../src/metadata.hh"
#include "../src/utils/cuda_err.cuh"
using std::cerr;
using std::cout;

using Data  = float;
using Quant = unsigned short;
Data*  data;
Quant* quant;
Data*  outlier;
auto   radius = 0;
auto   ebx2 = 1.0, ebx2_r = 1.0;
auto   unified_size = 512 * 512 * 512;

auto num_partitions = [](auto size, auto subsize) { return (size + subsize - 1) / subsize; };

__global__ void dummy() { float data = threadIdx.x; }

void Test1D(int n = 1)
{
    auto dimx = 512 * 512 * 512;

    static const auto Sequentiality = 8;
    static const auto DataSubsize   = MetadataTrait<1>::Block;
    auto              dim_block     = DataSubsize / Sequentiality;
    auto              dim_grid      = num_partitions(dimx, DataSubsize);

    for (auto i = 0; i < n; i++) {
        cout << "1Dc " << i << '\n';
        cusz::c_lorenzo_1d1l<Data, Quant, float, DataSubsize, Sequentiality><<<dim_grid, dim_block>>>  //
            (data, quant, dimx, radius, ebx2_r);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    for (auto i = 0; i < n; i++) {
        cout << "1Dx " << i << '\n';
        cusz::x_lorenzo_1d1l<Data, Quant><<<dim_grid, dim_block>>>  //
            (data, outlier, quant, dimx, radius, ebx2);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }
}

void Test2D(int n = 1)
{
    auto dimx = 512 * 32, dimy = 512 * 16;
    auto stridey = dimx;

    auto dim_block = dim3(16, 2);
    auto dim_grid  = dim3(
        num_partitions(dimx, 16),  //
        num_partitions(dimy, 16));

    for (auto i = 0; i < n; i++) {
        cout << "2Dc " << i << '\n';
        cusz::c_lorenzo_2d1l_16x16data_mapto16x2<Data, Quant, float><<<dim_grid, dim_block>>>  //
            (data, quant, dimx, dimy, stridey, radius, ebx2_r);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    for (auto i = 0; i < n; i++) {
        cout << "2Dx " << i << '\n';
        cusz::x_lorenzo_2d1l_16x16data_mapto16x2<Data, Quant><<<dim_grid, dim_block>>>  //
            (data, outlier, quant, dimx, dimy, stridey, radius, ebx2);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }
}

void Test3D(int n = 1)
{
    auto dimx = 512, dimy = 512, dimz = 512;
    auto stridey = 512, stridez = 512 * 512;

    auto dim_block = dim3(32, 1, 8);
    auto dim_grid  = dim3(
        num_partitions(dimx, 32),  //
        num_partitions(dimy, 8),   //
        num_partitions(dimz, 8)    //
    );

    for (auto i = 0; i < n; i++) {
        cout << "3Dc " << i << '\n';
        cusz::c_lorenzo_3d1l_32x8x8data_mapto32x1x8<Data, Quant><<<dim_grid, dim_block>>>  //
            (data, quant, dimx, dimy, dimz, stridey, stridez, radius, ebx2_r);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    for (auto i = 0; i < n; i++) {
        cout << "3Dx " << i << '\n';
        cusz::x_lorenzo_3d1l_32x8x8data_mapto32x1x8<<<dim_grid, dim_block>>>  //
            (data, outlier, quant, dimx, dimy, dimz, stridey, stridez, radius, ebx2);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }
}

int main(int argc, char** argv)
{
    cudaMallocManaged(&data, unified_size * sizeof(Data));
    cudaMallocManaged(&outlier, unified_size * sizeof(Data));
    cudaMallocManaged(&quant, unified_size * sizeof(Quant));

    Data* outlier = data;

    dummy<<<512, 512>>>();
    HANDLE_ERROR(cudaDeviceSynchronize());

    auto n = 1;
    if (argc > 0) n = atoi(argv[1]);

    Test1D(n);
    Test2D(n);
    Test3D(n);

    cudaFree(data);
    cudaFree(quant);
    cudaFree(outlier);

    cudaDeviceReset();

    return 0;
}
