/**
 * @file dummy_cxlorenzo.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-04-29
 * (repurposed) 2021-12-11
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include <iostream>
#include "../src/common.hh"
#include "../src/kernel/lorenzo.cuh"
#include "../src/kernel/lorenzo_prototype.cuh"
#include "../src/utils.hh"
using std::cerr;
using std::cout;

using Data  = float;
using Quant = unsigned short;

auto radius = 0;
auto ebx2 = 1.0, ebx2_r = 1.0;
auto unified_size = 512 * 512 * 512;

__global__ void dummy() { float data = threadIdx.x; }

void Test1D(Data* data, Quant* quant, int n = 1)
{
    auto dimx = 512 * 512 * 512;

    static const auto SEQ          = ChunkingTrait<1>::SEQ;
    static const auto DATA_SUBSIZE = ChunkingTrait<1>::BLOCK;
    auto              dim_block    = DATA_SUBSIZE / SEQ;
    auto              dim_grid     = ConfigHelper::get_npart(dimx, DATA_SUBSIZE);

    for (auto i = 0; i < n; i++) {
        cout << "1Dc " << i << '\n';
        cusz::c_lorenzo_1d1l<Data, Quant, float, DATA_SUBSIZE, SEQ><<<dim_grid, dim_block>>>  //
            (data, quant, dimx, radius, ebx2_r);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    for (auto i = 0; i < n; i++) {
        cout << "1Dx " << i << '\n';
        cusz::x_lorenzo_1d1l<Data, Quant><<<dim_grid, dim_block>>>  //
            (data, quant, dimx, radius, ebx2);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }
}

void Test2D(Data* data, Quant* quant, int n = 1)
{
    auto dimx = 512 * 32, dimy = 512 * 16;
    auto stridey = dimx;

    auto dim_block = dim3(16, 2);
    auto dim_grid  = dim3(
         ConfigHelper::get_npart(dimx, 16),  //
         ConfigHelper::get_npart(dimy, 16));

    for (auto i = 0; i < n; i++) {
        cout << "2Dc " << i << '\n';
        cusz::c_lorenzo_2d1l_16x16data_mapto16x2<Data, Quant, float><<<dim_grid, dim_block>>>  //
            (data, quant, dimx, dimy, stridey, radius, ebx2_r);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    for (auto i = 0; i < n; i++) {
        cout << "2Dx " << i << '\n';
        cusz::x_lorenzo_2d1l_16x16data_mapto16x2<Data, Quant><<<dim_grid, dim_block>>>  //
            (data, quant, dimx, dimy, stridey, radius, ebx2);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }
}

void Test3D(Data* data, Quant* quant, int n = 1)
{
    auto dimx = 512, dimy = 512, dimz = 512;
    auto stridey = 512, stridez = 512 * 512;

    auto dim_block = dim3(32, 1, 8);
    auto dim_grid  = dim3(
         ConfigHelper::get_npart(dimx, 32),  //
         ConfigHelper::get_npart(dimy, 8),   //
         ConfigHelper::get_npart(dimz, 8)    //
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
            (data, quant, dimx, dimy, dimz, stridey, stridez, radius, ebx2);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }
}

int main(int argc, char** argv)
{
    Data*  data;
    Quant* quant;

    cudaMallocManaged(&data, unified_size * sizeof(Data));
    cudaMallocManaged(&quant, unified_size * sizeof(Quant));

    dummy<<<512, 512>>>();
    HANDLE_ERROR(cudaDeviceSynchronize());

    auto n = 1;
    if (argc > 1) n = atoi(argv[1]);

    Test1D(data, quant, n);
    Test2D(data, quant, n);
    Test3D(data, quant, n);

    cudaFree(data);
    cudaFree(quant);

    cudaDeviceReset();

    return 0;
}
