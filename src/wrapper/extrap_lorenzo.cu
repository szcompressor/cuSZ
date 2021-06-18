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

#include <numeric>
#include <stdexcept>
#include "../gather_scatter.cuh"
#include "../kernel/lorenzo.h"
#include "extrap_lorenzo.h"

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

auto get_npart = [](auto size, auto subsize) {
    static_assert(
        std::numeric_limits<decltype(size)>::is_integer and std::numeric_limits<decltype(subsize)>::is_integer,
        "[get_npart] must be plain interger types.");
    return (size + subsize - 1) / subsize;
};
auto get_len_from_dim3 = [](dim3 size) { return size.x * size.y * size.z; };
auto get_stride3       = [](dim3 size) -> dim3 { return dim3(1, size.x, size.x * size.y); };

}  // namespace

template <typename Data, typename Quant, typename FP>
void dryrun_lorenzo(Data* data, dim3 size3, int ndim, FP eb)
{
    // TODO
}

template <typename Data, typename Quant, typename FP, bool DELAY_POSTQUANT>
void compress_lorenzo_construct(Data* data, Quant* quant, dim3 size3, int ndim, FP eb, int radius)
{
    FP   ebx2_r  = 1 / (eb * 2);
    auto stride3 = get_stride3(size3);

    if (DELAY_POSTQUANT == (quant == nullptr)) {
        throw std::runtime_error("[compress_lorenzo_construct] delaying postquant <=> (var quant is null)");
    }

    if (ndim == 1) {
        constexpr auto SEQ       = 4;
        constexpr auto SUBSIZE   = 256;
        auto           dim_block = SUBSIZE / SEQ;
        auto           dim_grid  = get_npart(size3.x, SUBSIZE);
        cusz::c_lorenzo_1d1l<Data, Quant, FP, SUBSIZE, SEQ, DELAY_POSTQUANT><<<dim_grid, dim_block>>>  //
            (data, quant, size3.x, radius, ebx2_r);
    }
    else if (ndim == 2) {  // y-sequentiality == 8
        auto dim_block = dim3(16, 2);
        auto dim_grid  = dim3(get_npart(size3.x, 16), get_npart(size3.y, 16));
        cusz::c_lorenzo_2d1l_16x16data_mapto16x2<Data, Quant, FP><<<dim_grid, dim_block>>>  //
            (data, quant, size3.x, size3.y, stride3.y, radius, ebx2_r);
    }
    else if (ndim == 3) {  // y-sequentiality == 8
        auto dim_block = dim3(32, 1, 8);
        auto dim_grid  = dim3(get_npart(size3.x, 32), get_npart(size3.y, 8), get_npart(size3.z, 8));
        cusz::c_lorenzo_3d1l_32x8x8data_mapto32x1x8<Data, Quant, FP>
            <<<dim_grid, dim_block>>>(data, quant, size3.x, size3.y, size3.z, stride3.y, stride3.z, radius, ebx2_r);
    }
    cudaDeviceSynchronize();
}

/* specialize */
template <>
void compress_lorenzo_construct<float, unsigned short, float, false>  //
    (float*, unsigned short*, dim3, int, float, int);

/********************************************************************************
 * decompression
 ********************************************************************************/

template <typename Data, typename Quant, typename FP, bool DELAY_POSTQUANT>
void decompress_lorenzo_reconstruct(Data* xdata, Quant* quant, dim3 size3, int ndim, FP eb, int radius)
{
    auto stride3 = get_stride3(size3);
    auto ebx2    = eb * 2;

    if (ndim == 1) {  // y-sequentiality == 8
        constexpr auto SEQ       = 8;
        constexpr auto SUBSIZE   = 256;
        auto           dim_block = SUBSIZE / SEQ;
        auto           dim_grid  = get_npart(size3.x, SUBSIZE);

        cusz::x_lorenzo_1d1l<Data, Quant, FP, SEQ, DELAY_POSTQUANT><<<dim_grid, dim_block>>>  //
            (xdata, quant, size3.x, radius, ebx2);
    }
    else if (ndim == 2) {  // y-sequentiality == 8

        auto dim_block = dim3(16, 2);
        auto dim_grid  = dim3(get_npart(size3.x, 16), get_npart(size3.y, 16));

        cusz::x_lorenzo_2d1l_16x16data_mapto16x2<Data, Quant, FP, DELAY_POSTQUANT><<<dim_grid, dim_block>>>  //
            (xdata, quant, size3.x, size3.y, stride3.y, radius, ebx2);
    }
    else if (ndim == 3) {  // y-sequentiality == 8

        auto dim_block = dim3(32, 1, 8);
        auto dim_grid  = dim3(get_npart(size3.x, 32), get_npart(size3.y, 8), get_npart(size3.z, 8));

        cusz::x_lorenzo_3d1l_32x8x8data_mapto32x1x8<Data, Quant, FP, DELAY_POSTQUANT><<<dim_grid, dim_block>>>  //
            (xdata, quant, size3.x, size3.y, size3.z, stride3.y, stride3.z, radius, ebx2);
    }
    cudaDeviceSynchronize();
}

template <>
void decompress_lorenzo_reconstruct<float, unsigned short, float, false>  //
    (float*, unsigned short*, dim3, int, float, int);
