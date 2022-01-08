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
#include "common.hh"
#include "kernel/lorenzo.cuh"
#include "kernel/lorenzo_prototype.cuh"
#include "utils.hh"
using std::cerr;
using std::cout;

using Data  = float;
using Quant = unsigned short;

auto radius = 0;
auto ebx2 = 1.0, ebx2_r = 1.0;
auto unified_size = 512 * 512 * 512;

__global__ void dummy() { float data = threadIdx.x; }

cudaStream_t stream1;
cudaStream_t stream2;
cudaStream_t stream3;

void Test1D(Data* data, Quant* quant)
{
    auto dimx    = 512 * 512 * 512;
    auto outlier = data;

    static const auto SEQ          = ChunkingTrait<1>::SEQ;
    static const auto DATA_SUBSIZE = ChunkingTrait<1>::BLOCK;
    auto              dim_block    = DATA_SUBSIZE / SEQ;
    auto              dim_grid     = ConfigHelper::get_npart(dimx, DATA_SUBSIZE);

    cusz::c_lorenzo_1d1l<Data, Quant, float, DATA_SUBSIZE, SEQ><<<dim_grid, dim_block, 0, stream1>>>  //
        (data, quant, outlier, dimx, radius, ebx2_r);

    cusz::x_lorenzo_1d1l<Data, Quant><<<dim_grid, dim_block, 0, stream1>>>  //
        (outlier, quant, data, dimx, radius, ebx2);
}

void Test2D(Data* data, Quant* quant)
{
    auto dimx = 512 * 32, dimy = 512 * 16;
    auto stridey = dimx;
    auto outlier = data;

    auto dim_block = dim3(16, 2);
    auto dim_grid  = dim3(
         ConfigHelper::get_npart(dimx, 16),  //
         ConfigHelper::get_npart(dimy, 16));

    cusz::c_lorenzo_2d1l_16x16data_mapto16x2<Data, Quant, float><<<dim_grid, dim_block, 0, stream2>>>  //
        (data, quant, outlier, dimx, dimy, stridey, radius, ebx2_r);

    cusz::x_lorenzo_2d1l_16x16data_mapto16x2<Data, Quant><<<dim_grid, dim_block, 0, stream2>>>  //
        (outlier, quant, data, dimx, dimy, stridey, radius, ebx2);
}

void Test3D(Data* data, Quant* quant)
{
    auto dimx = 512, dimy = 512, dimz = 512;
    auto stridey = 512, stridez = 512 * 512;
    auto outlier = data;

    auto dim_block = dim3(32, 1, 8);
    auto dim_grid  = dim3(
         ConfigHelper::get_npart(dimx, 32),  //
         ConfigHelper::get_npart(dimy, 8),   //
         ConfigHelper::get_npart(dimz, 8)    //
     );

    cusz::c_lorenzo_3d1l_32x8x8data_mapto32x1x8<Data, Quant><<<dim_grid, dim_block, 0, stream3>>>  //
        (data, quant, outlier, dimx, dimy, dimz, stridey, stridez, radius, ebx2_r);

    cusz::x_lorenzo_3d1l_32x8x8data_mapto32x1x8<<<dim_grid, dim_block, 0, stream3>>>  //
        (outlier, quant, data, dimx, dimy, dimz, stridey, stridez, radius, ebx2);
}

int main(int argc, char** argv)
{
    auto n           = 1;
    bool multistream = false;
    if (argc == 3) {
        n = atoi(argv[1]);
        if (atoi(argv[2]) == 1) {
            multistream = true;
            cout << "use multistream\n";
        }
    }

    Data*  data;
    Quant* quant;

    cudaMallocManaged(&data, unified_size * sizeof(Data));
    cudaMallocManaged(&quant, unified_size * sizeof(Quant));

    if (not multistream) {
        cudaStreamCreate(&stream1);
        stream2 = stream1;
        stream3 = stream1;
    }
    else {
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        cudaStreamCreate(&stream3);
    }

    dummy<<<512, 512, 0, stream1>>>();
    HANDLE_ERROR(cudaStreamSynchronize(stream1));

    auto t1 = hires::now();

    for (auto i = 0; i < n; i++) {
        Test1D(data, quant);
        Test2D(data, quant);
        Test3D(data, quant);
    }

    if (not multistream) {  //
        HANDLE_ERROR(cudaStreamSynchronize(stream1));
    }
    else {
        HANDLE_ERROR(cudaStreamSynchronize(stream1));
        HANDLE_ERROR(cudaStreamSynchronize(stream2));
        HANDLE_ERROR(cudaStreamSynchronize(stream3));
    }

    auto t2 = hires::now();
    cout << "time elapsed:\t" << static_cast<duration_t>(t2 - t1).count() << '\n';

    cudaFree(data);
    cudaFree(quant);

    if (stream1) cudaStreamDestroy(stream1);
    if (stream2) cudaStreamDestroy(stream2);
    if (stream3) cudaStreamDestroy(stream3);

    cudaDeviceReset();

    return 0;
}
