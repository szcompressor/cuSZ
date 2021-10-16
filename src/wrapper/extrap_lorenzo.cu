/**
 * @file extrap_lorenzo.cu
 * @author Jiannan Tian
 * @brief A high-level LorenzoND wrapper. Allocations are explicitly out of called functions.
 * @version 0.3
 * @date 2021-06-16
 * (rev.1) 2021-09-18
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "../common.hh"
#include "../utils.hh"

#include "extrap_lorenzo.cuh"

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

}  // namespace

template <typename T, typename E, typename FP>
cusz::PredictorLorenzo<T, E, FP>::PredictorLorenzo(dim3 xyz, double _eb, int _radius, bool _delay_postquant)
{
    // error bound
    eb     = _eb;
    ebx2   = eb * 2;
    ebx2_r = 1 / ebx2;

    // size
    size      = xyz;
    leap      = dim3(1, size.x, size.x * size.y);
    len_data  = size.x * size.y * size.z;
    len_quant = len_data;
    radius    = _radius;

    ndim = 3;
    if (size.z == 1) ndim = 2;
    if (size.z == 1 and size.y == 1) ndim = 1;

    // on off
    delay_postquant = _delay_postquant;
}

template <typename T, typename E, typename FP>
template <bool DELAY_POSTQUANT>
void cusz::PredictorLorenzo<T, E, FP>::construct_proxy(T* data, T* anchor, E* errctrl)
{
    // TODO put into conditional compile
    auto timer = new cuda_timer_t;
    timer->timer_start();

    if (ndim == 1) {
        constexpr auto SEQ          = 4;
        constexpr auto DATA_SUBSIZE = 256;
        auto           dim_block    = DATA_SUBSIZE / SEQ;
        auto           dim_grid     = ConfigHelper::get_npart(size.x, DATA_SUBSIZE);
        cusz::c_lorenzo_1d1l<T, E, FP, DATA_SUBSIZE, SEQ, DELAY_POSTQUANT><<<dim_grid, dim_block>>>  //
            (data, errctrl, size.x, radius, ebx2_r);
    }
    else if (ndim == 2) {  // y-sequentiality == 8
        auto dim_block = dim3(16, 2);
        auto dim_grid  = dim3(ConfigHelper::get_npart(size.x, 16), ConfigHelper::get_npart(size.y, 16));
        cusz::c_lorenzo_2d1l_16x16data_mapto16x2<T, E, FP><<<dim_grid, dim_block>>>  //
            (data, errctrl, size.x, size.y, leap.y, radius, ebx2_r);
    }
    else if (ndim == 3) {  // y-sequentiality == 8
        auto dim_block = dim3(32, 1, 8);
        auto dim_grid  = dim3(
            ConfigHelper::get_npart(size.x, 32), ConfigHelper::get_npart(size.y, 8),
            ConfigHelper::get_npart(size.z, 8));
        cusz::c_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, E, FP>
            <<<dim_grid, dim_block>>>(data, errctrl, size.x, size.y, size.z, leap.y, leap.z, radius, ebx2_r);
    }
    else {
        throw std::runtime_error("Lorenzo only works for 123-D.");
    }

    time_elapsed = timer->timer_end_get_elapsed_time();
    CHECK_CUDA(cudaDeviceSynchronize());
    delete timer;
}

template <typename T, typename E, typename FP>
template <bool DELAY_POSTQUANT>
void cusz::PredictorLorenzo<T, E, FP>::reconstruct_proxy(T* anchor, E* errctrl, T* xdata)
{
    auto timer = new cuda_timer_t;
    timer->timer_start();

    if (ndim == 1) {  // y-sequentiality == 8
        constexpr auto SEQ          = 8;
        constexpr auto DATA_SUBSIZE = 256;
        auto           dim_block    = DATA_SUBSIZE / SEQ;
        auto           dim_grid     = ConfigHelper::get_npart(size.x, DATA_SUBSIZE);
        cusz::x_lorenzo_1d1l<T, E, FP, DATA_SUBSIZE, SEQ, DELAY_POSTQUANT><<<dim_grid, dim_block>>>  //
            (xdata, errctrl, size.x, radius, ebx2);
    }
    else if (ndim == 2) {  // y-sequentiality == 8
        auto dim_block = dim3(16, 2);
        auto dim_grid  = dim3(ConfigHelper::get_npart(size.x, 16), ConfigHelper::get_npart(size.y, 16));

        cusz::x_lorenzo_2d1l_16x16data_mapto16x2<T, E, FP, DELAY_POSTQUANT><<<dim_grid, dim_block>>>  //
            (xdata, errctrl, size.x, size.y, leap.y, radius, ebx2);
    }
    else if (ndim == 3) {  // y-sequentiality == 8
        auto dim_block = dim3(32, 1, 8);
        auto dim_grid  = dim3(
            ConfigHelper::get_npart(size.x, 32), ConfigHelper::get_npart(size.y, 8),
            ConfigHelper::get_npart(size.z, 8));

        cusz::x_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, E, FP, DELAY_POSTQUANT><<<dim_grid, dim_block>>>  //
            (xdata, errctrl, size.x, size.y, size.z, leap.y, leap.z, radius, ebx2);
    }

    time_elapsed = timer->timer_end_get_elapsed_time();
    CHECK_CUDA(cudaDeviceSynchronize());
    delete timer;
}

template <typename T, typename E, typename FP>
void cusz::PredictorLorenzo<T, E, FP>::construct(T* in_data, T* out_anchor, E* out_errctrl)
{
    if (not delay_postquant)
        construct_proxy<false>(in_data, out_anchor, out_errctrl);
    else
        throw std::runtime_error("construct_proxy<delay_postquant==true> not implemented.");
}

template <typename T, typename E, typename FP>
void cusz::PredictorLorenzo<T, E, FP>::reconstruct(T* in_anchor, E* in_errctrl, T* out_xdata)
{
    if (not delay_postquant)
        reconstruct_proxy<false>(in_anchor, in_errctrl, out_xdata);
    else
        throw std::runtime_error("construct_proxy<delay_postquant==true> not implemented.");
}

template class cusz::PredictorLorenzo<float, uint16_t, float>;