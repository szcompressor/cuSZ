/**
 * @file extrap_lorenzo.cu
 * @author Jiannan Tian
 * @brief A high-level LorenzoND wrapper. Allocations are explicitly out of called functions.
 * @version 0.3
 * @date 2021-06-16
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include <iostream>
#include <stdexcept>
#include "extrap_lorenzo.cuh"

#include "../type_trait.hh"
#include "../utils/timer.hh"

#ifdef DPCPP_SHOWCASE
#include "../kernel/lorenzo_prototype.cuh"

using cusz::prototype::c_lorenzo_1d1l;
using cusz::prototype::c_lorenzo_2d1l;
using cusz::prototype::c_lorenzo_3d1l;
using cusz::prototype::x_lorenzo_1d1l;
using cusz::prototype::x_lorenzo_2d1l;
using cusz::prototype::x_lorenzo_3d1l;

#else
#include "../kernel/lorenzo.cuh"
#endif

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

namespace {

#ifndef __CUDACC__
struct __dim3_compat {
    unsigned int x, y, z;
    __dim3_compat(unsigned int _x, unsigned int _y, unsigned int _z){};
};

using dim3 = __dim3_compat;
#endif

auto get_stride3 = [](dim3 size) -> dim3 { return dim3(1, size.x, size.x * size.y); };

}  // namespace

template <typename Data, typename Quant, typename FP>
void dryrun_lorenzo(Data* data, dim3 size3, int ndim, FP eb)
{
    // TODO
}

template <typename Data, typename Quant, typename FP, bool DELAY_POSTQUANT>
void compress_lorenzo_construct(Data* data, Quant* quant, dim3 size3, int ndim, FP eb, int radius, float& ms)
{
    FP   ebx2_r  = 1 / (eb * 2);
    auto stride3 = get_stride3(size3);

    if (DELAY_POSTQUANT != (quant == nullptr)) {
        throw std::runtime_error("[compress_lorenzo_construct] delaying postquant <=> (var quant is null)");
    }

#ifdef DPCPP_SHOWCASE

    std::cout << "DPCPP compression showcase, revert to prototype ND kernel(s)\n";
    if (ndim == 1) {
        constexpr auto DATA_SUBSIZE = 256;
        auto           dim_block    = DATA_SUBSIZE;
        auto           dim_grid     = ConfigHelper::get_npart(size3.x, DATA_SUBSIZE);
        c_lorenzo_1d1l<Data, Quant, FP, DATA_SUBSIZE><<<dim_grid, dim_block>>>  //
            (data, quant, size3.x, radius, ebx2_r, nullptr, nullptr, 1.0);
    }
    else if (ndim == 2) {
        constexpr auto DATA_SUBSIZE = 16;
        auto           dim_block    = dim3(DATA_SUBSIZE, DATA_SUBSIZE);
        auto           dim_grid =
            dim3(ConfigHelper::get_npart(size3.x, DATA_SUBSIZE), ConfigHelper::get_npart(size3.y, DATA_SUBSIZE));
        c_lorenzo_2d1l<Data, Quant, FP, DATA_SUBSIZE><<<dim_grid, dim_block>>>  //
            (data, quant, size3.x, size3.y, size3.x, radius, ebx2_r, nullptr, nullptr, 1.0);
    }
    else if (ndim == 3) {
        constexpr auto DATA_SUBSIZE = 8;
        auto           dim_block    = dim3(DATA_SUBSIZE, DATA_SUBSIZE, DATA_SUBSIZE);
        auto           dim_grid     = dim3(
            ConfigHelper::get_npart(size3.x, DATA_SUBSIZE),  //
            ConfigHelper::get_npart(size3.y, DATA_SUBSIZE),  //
            ConfigHelper::get_npart(size3.z, DATA_SUBSIZE));
        c_lorenzo_3d1l<Data, Quant, FP, DATA_SUBSIZE><<<dim_grid, dim_block>>>  //
            (data, quant, size3.x, size3.y, size3.z, size3.x, size3.x * size3.y, radius, ebx2_r, nullptr, nullptr, 1.0);
    }

#else

    // TODO put into conditional compile
    auto timer_lorenzo = new cuda_timer_t;
    timer_lorenzo->timer_start();

    if (ndim == 1) {
        constexpr auto SEQ          = 4;
        constexpr auto DATA_SUBSIZE = 256;
        auto           dim_block    = DATA_SUBSIZE / SEQ;
        auto           dim_grid     = ConfigHelper::get_npart(size3.x, DATA_SUBSIZE);
        cusz::c_lorenzo_1d1l<Data, Quant, FP, DATA_SUBSIZE, SEQ, DELAY_POSTQUANT><<<dim_grid, dim_block>>>  //
            (data, quant, size3.x, radius, ebx2_r);
    }
    else if (ndim == 2) {  // y-sequentiality == 8
        auto dim_block = dim3(16, 2);
        auto dim_grid  = dim3(ConfigHelper::get_npart(size3.x, 16), ConfigHelper::get_npart(size3.y, 16));
        cusz::c_lorenzo_2d1l_16x16data_mapto16x2<Data, Quant, FP><<<dim_grid, dim_block>>>  //
            (data, quant, size3.x, size3.y, stride3.y, radius, ebx2_r);
    }
    else if (ndim == 3) {  // y-sequentiality == 8
        auto dim_block = dim3(32, 1, 8);
        auto dim_grid  = dim3(
            ConfigHelper::get_npart(size3.x, 32), ConfigHelper::get_npart(size3.y, 8),
            ConfigHelper::get_npart(size3.z, 8));
        cusz::c_lorenzo_3d1l_32x8x8data_mapto32x1x8<Data, Quant, FP>
            <<<dim_grid, dim_block>>>(data, quant, size3.x, size3.y, size3.z, stride3.y, stride3.z, radius, ebx2_r);
    }
#endif

    ms = timer_lorenzo->timer_end_get_elapsed_time();
    cudaDeviceSynchronize();

    delete timer_lorenzo;
}

/********************************************************************************
 * decompression
 ********************************************************************************/

template <typename Data, typename Quant, typename FP, bool DELAY_POSTQUANT>
void decompress_lorenzo_reconstruct(Data* xdata, Quant* quant, dim3 size3, int ndim, FP eb, int radius, float& ms)
{
    auto stride3 = get_stride3(size3);
    auto ebx2    = eb * 2;

#ifdef DPCPP_SHOWCASE

    std::cout << "DPCPP decompression showcase, revert to prototype ND kernel(s)\n";
    if (ndim == 1) {  // y-sequentiality == 8
        constexpr auto DATA_SUBSIZE = 256;
        auto           dim_block    = DATA_SUBSIZE;
        auto           dim_grid     = ConfigHelper::get_npart(size3.x, DATA_SUBSIZE);
        // TODO delete outlier-data
        x_lorenzo_1d1l<Data, Quant, FP, DATA_SUBSIZE><<<dim_grid, dim_block>>>  //
            (xdata, quant, size3.x, radius, ebx2);
    }
    else if (ndim == 2) {
        constexpr auto DATA_SUBSIZE = 16;
        auto           dim_block    = dim3(DATA_SUBSIZE, DATA_SUBSIZE);
        auto           dim_grid =
            dim3(ConfigHelper::get_npart(size3.x, DATA_SUBSIZE), ConfigHelper::get_npart(size3.y, DATA_SUBSIZE));
        x_lorenzo_2d1l<Data, Quant, FP, DATA_SUBSIZE><<<dim_grid, dim_block>>>  //
            (xdata, quant, size3.x, size3.y, size3.x, radius, ebx2);
    }
    else if (ndim == 3) {
        constexpr auto DATA_SUBSIZE = 8;
        auto           dim_block    = dim3(DATA_SUBSIZE, DATA_SUBSIZE, DATA_SUBSIZE);
        auto           dim_grid     = dim3(
            ConfigHelper::get_npart(size3.x, DATA_SUBSIZE),  //
            ConfigHelper::get_npart(size3.y, DATA_SUBSIZE),  //
            ConfigHelper::get_npart(size3.z, DATA_SUBSIZE));
        x_lorenzo_3d1l<Data, Quant, FP, DATA_SUBSIZE><<<dim_grid, dim_block>>>  //
            (xdata, quant, size3.x, size3.y, size3.z, size3.x, size3.x * size3.y, radius, ebx2);
    }

#else

    auto timer_lorenzo = new cuda_timer_t;
    timer_lorenzo->timer_start();

    if (ndim == 1) {  // y-sequentiality == 8
        constexpr auto SEQ          = 8;
        constexpr auto DATA_SUBSIZE = 256;
        auto           dim_block    = DATA_SUBSIZE / SEQ;
        auto           dim_grid     = ConfigHelper::get_npart(size3.x, DATA_SUBSIZE);
        cusz::x_lorenzo_1d1l<Data, Quant, FP, DATA_SUBSIZE, SEQ, DELAY_POSTQUANT><<<dim_grid, dim_block>>>  //
            (xdata, quant, size3.x, radius, ebx2);
    }
    else if (ndim == 2) {  // y-sequentiality == 8

        auto dim_block = dim3(16, 2);
        auto dim_grid  = dim3(ConfigHelper::get_npart(size3.x, 16), ConfigHelper::get_npart(size3.y, 16));

        cusz::x_lorenzo_2d1l_16x16data_mapto16x2<Data, Quant, FP, DELAY_POSTQUANT><<<dim_grid, dim_block>>>  //
            (xdata, quant, size3.x, size3.y, stride3.y, radius, ebx2);
    }
    else if (ndim == 3) {  // y-sequentiality == 8

        auto dim_block = dim3(32, 1, 8);
        auto dim_grid  = dim3(
            ConfigHelper::get_npart(size3.x, 32), ConfigHelper::get_npart(size3.y, 8),
            ConfigHelper::get_npart(size3.z, 8));

        cusz::x_lorenzo_3d1l_32x8x8data_mapto32x1x8<Data, Quant, FP, DELAY_POSTQUANT><<<dim_grid, dim_block>>>  //
            (xdata, quant, size3.x, size3.y, size3.z, stride3.y, stride3.z, radius, ebx2);
    }
#endif

    ms = timer_lorenzo->timer_end_get_elapsed_time();
    cudaDeviceSynchronize();
    delete timer_lorenzo;
}

/* instantiate */
#define INSTANTIATE_COMPRESS_LORENZO_CONSTRUCT(Data, Quant, FP) \
    template void compress_lorenzo_construct<Data, Quant, FP, false>(Data*, Quant*, dim3, int, FP, int, float&);

INSTANTIATE_COMPRESS_LORENZO_CONSTRUCT(float, uint8_t, float)
INSTANTIATE_COMPRESS_LORENZO_CONSTRUCT(float, uint8_t, double)
INSTANTIATE_COMPRESS_LORENZO_CONSTRUCT(float, uint16_t, float)
INSTANTIATE_COMPRESS_LORENZO_CONSTRUCT(float, uint16_t, double)
INSTANTIATE_COMPRESS_LORENZO_CONSTRUCT(float, uint32_t, float)
INSTANTIATE_COMPRESS_LORENZO_CONSTRUCT(float, uint32_t, double)
// INSTANTIATE_COMPRESS_LORENZO_CONSTRUCT(double, uint8_t, float)
// INSTANTIATE_COMPRESS_LORENZO_CONSTRUCT(double, uint8_t, double)
// INSTANTIATE_COMPRESS_LORENZO_CONSTRUCT(double, uint16_t, float)
// INSTANTIATE_COMPRESS_LORENZO_CONSTRUCT(double, uint16_t, double)

#define INSTANTIATE_DECOMPRESS_LORENZO_RECONSTRUCT(Data, Quant, FP) \
    template void decompress_lorenzo_reconstruct<Data, Quant, FP, false>(Data*, Quant*, dim3, int, FP, int, float&);

INSTANTIATE_DECOMPRESS_LORENZO_RECONSTRUCT(float, uint8_t, float)
INSTANTIATE_DECOMPRESS_LORENZO_RECONSTRUCT(float, uint8_t, double)
INSTANTIATE_DECOMPRESS_LORENZO_RECONSTRUCT(float, uint16_t, float)
INSTANTIATE_DECOMPRESS_LORENZO_RECONSTRUCT(float, uint16_t, double)
INSTANTIATE_DECOMPRESS_LORENZO_RECONSTRUCT(float, uint32_t, float)
INSTANTIATE_DECOMPRESS_LORENZO_RECONSTRUCT(float, uint32_t, double)
// INSTANTIATE_DECOMPRESS_LORENZO_RECONSTRUCT(double, uint8_t, float)
// INSTANTIATE_DECOMPRESS_LORENZO_RECONSTRUCT(double, uint8_t, double)
// INSTANTIATE_DECOMPRESS_LORENZO_RECONSTRUCT(double, uint16_t, float)
// INSTANTIATE_DECOMPRESS_LORENZO_RECONSTRUCT(double, uint16_t, double)
