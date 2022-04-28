/**
 * @file extrap_lorenzo.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-06-16
 * (rev.1) 2021-09-18 (rev.2) 2022-01-10
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_COMPONENT_EXTRAP_LORENZO_CUH
#define CUSZ_COMPONENT_EXTRAP_LORENZO_CUH

#include <clocale>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "../common.hh"
#include "../component/predictor.hh"
#include "../utils.hh"

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

#define ALLOCDEV(VAR, SYM, NBYTE)                    \
    if (NBYTE != 0) {                                \
        CHECK_CUDA(cudaMalloc(&d_##VAR, NBYTE));     \
        CHECK_CUDA(cudaMemset(d_##VAR, 0x0, NBYTE)); \
    }

#define ALLOCDEV2(VAR, TYPE, LEN)                                 \
    if (LEN != 0) {                                               \
        CHECK_CUDA(cudaMalloc(&d_##VAR, sizeof(TYPE) * LEN));     \
        CHECK_CUDA(cudaMemset(d_##VAR, 0x0, sizeof(TYPE) * LEN)); \
    }

#define FREE_DEV_ARRAY(VAR)            \
    if (d_##VAR) {                     \
        CHECK_CUDA(cudaFree(d_##VAR)); \
        d_##VAR = nullptr;             \
    }

#define DEFINE_ARRAY(VAR, TYPE) TYPE* d_##VAR{nullptr};

namespace cusz {

template <typename T, typename E, typename FP>
PredictorLorenzo<T, E, FP>::impl::impl()
{
}

template <typename T, typename E, typename FP>
PredictorLorenzo<T, E, FP>::impl::~impl()
{
    FREE_DEV_ARRAY(anchor);
    FREE_DEV_ARRAY(errctrl);
}

template <typename T, typename E, typename FP>
void PredictorLorenzo<T, E, FP>::impl::clear_buffer()
{
    cudaMemset(d_errctrl, 0x0, sizeof(E) * this->rtlen.assigned.quant);
}

template <typename T, typename E, typename FP>
void PredictorLorenzo<T, E, FP>::impl::init(size_t x, size_t y, size_t z, bool dbg_print)
{
    auto len3 = dim3(x, y, z);
    init(len3, dbg_print);
}

template <typename T, typename E, typename FP>
void PredictorLorenzo<T, E, FP>::impl::init(dim3 xyz, bool dbg_print)
{
    this->derive_alloclen(xyz);

    // allocate
    ALLOCDEV2(anchor, T, this->alloclen.assigned.anchor);
    ALLOCDEV2(errctrl, E, this->alloclen.assigned.quant);
    if (not outlier_overlapped) ALLOCDEV(outlier, T, this->alloclen.assigned.outlier);

    if (dbg_print) this->debug_list_alloclen<T, E, FP>();
}

template <typename T, typename E, typename FP>
E* PredictorLorenzo<T, E, FP>::impl::expose_quant() const
{
    return d_errctrl;
}
template <typename T, typename E, typename FP>
E* PredictorLorenzo<T, E, FP>::impl::expose_errctrl() const
{
    return d_errctrl;
}
template <typename T, typename E, typename FP>
T* PredictorLorenzo<T, E, FP>::impl::expose_anchor() const
{
    return d_anchor;
}
template <typename T, typename E, typename FP>
T* PredictorLorenzo<T, E, FP>::impl::expose_outlier() const
{
    return d_outlier;
}

template <typename T, typename E, typename FP>
void PredictorLorenzo<T, E, FP>::impl::construct(
    dim3 const len3,
    T* __restrict__ in_data,
    T*& out_anchor,
    E*& out_errctrl,
    T*& __restrict__ out_outlier,
    double const eb,
    int const    radius,
    cudaStream_t stream)
{
    out_anchor  = d_anchor;
    out_errctrl = d_errctrl;
    out_outlier = d_outlier;

    this->derive_rtlen(len3);

    if (not delay_postquant)
        construct_proxy<false>(in_data, out_anchor, out_errctrl, out_outlier, eb, radius, stream);
    else
        throw std::runtime_error("construct_proxy<delay_postquant==true> not implemented.");
}

template <typename T, typename E, typename FP>
void PredictorLorenzo<T, E, FP>::impl::reconstruct(
    dim3 len3,
    T* __restrict__ in_outlier,
    T* in_anchor,
    E* in_errctrl,
    T*& __restrict__ out_xdata,
    double const eb,
    int const    radius,
    cudaStream_t stream)
{
    this->derive_rtlen(len3);

    if (not delay_postquant)
        reconstruct_proxy<false>(in_outlier, in_anchor, in_errctrl, out_xdata, eb, radius, stream);
    else
        throw std::runtime_error("construct_proxy<delay_postquant==true> not implemented.");
}

template <typename T, typename E, typename FP>
void PredictorLorenzo<T, E, FP>::impl::construct(
    dim3 const   len3,
    T*           in_data__out_outlier,
    T*&          out_anchor,
    E*&          out_errctrl,
    double const eb,
    int const    radius,
    cudaStream_t stream)
{
    out_anchor  = d_anchor;
    out_errctrl = d_errctrl;

    derive_rtlen(len3);
    this->check_rtlen();

    if (not delay_postquant)
        construct_proxy<false>(in_data__out_outlier, out_anchor, out_errctrl, nullptr, eb, radius, stream);
    else
        throw std::runtime_error("construct_proxy<delay_postquant==true> not implemented.");
}

template <typename T, typename E, typename FP>
void PredictorLorenzo<T, E, FP>::impl::reconstruct(
    dim3         len3,
    T*&          in_outlier__out_xdata,
    T*           in_anchor,
    E*           in_errctrl,
    double const eb,
    int const    radius,
    cudaStream_t stream)
{
    derive_rtlen(len3);
    this->check_rtlen();

    if (not delay_postquant)
        reconstruct_proxy<false>(nullptr, in_anchor, in_errctrl, in_outlier__out_xdata, eb, radius, stream);
    else
        throw std::runtime_error("construct_proxy<delay_postquant==true> not implemented.");
}

template <typename T, typename E, typename FP>
float PredictorLorenzo<T, E, FP>::impl::get_time_elapsed() const
{
    return time_elapsed;
}

template <typename T, typename E, typename FP>
template <bool DELAY_POSTQUANT>
void PredictorLorenzo<T, E, FP>::impl::construct_proxy(
    T* const in_data,
    T* const out_anchor,
    E* const out_errctrl,
    T* const __restrict__ __out_outlier,
    double const eb,
    int const    radius,
    cudaStream_t stream)
{
    // error bound
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;

    // decide if destructive for the input (data)
    auto out_outlier = __out_outlier == nullptr ? in_data : __out_outlier;

    // TODO put into conditional compile
    cuda_timer_t timer;
    timer.timer_start(stream);

    if (get_ndim() == 1) {
        constexpr auto SEQ         = 4;
        constexpr auto DATA_SUBLEN = 256;
        auto           dim_block   = DATA_SUBLEN / SEQ;
        auto           dim_grid    = ConfigHelper::get_npart(get_x(), DATA_SUBLEN);

        ::cusz::c_lorenzo_1d1l<T, E, FP, DATA_SUBLEN, SEQ, DELAY_POSTQUANT>  //
            <<<dim_grid, dim_block, 0, stream>>>                             //
            (in_data, out_errctrl, out_outlier, get_x(), radius, ebx2_r);
    }
    else if (get_ndim() == 2) {  // y-sequentiality == 8
        auto dim_block = dim3(16, 2);
        auto dim_grid  = ConfigHelper::get_pardeg3(this->get_len3(), {16, 16, 1});

        ::cusz::c_lorenzo_2d1l_16x16data_mapto16x2<T, E, FP>  //
            <<<dim_grid, dim_block, 0, stream>>>              //
            (in_data, out_errctrl, out_outlier, get_x(), get_y(), get_leap().y, radius, ebx2_r);
    }
    else if (get_ndim() == 3) {  // y-sequentiality == 8
        auto dim_block = dim3(32, 1, 8);
        auto dim_grid  = ConfigHelper::get_pardeg3(this->get_len3(), {32, 8, 8});

        ::cusz::c_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, E, FP>  //
            <<<dim_grid, dim_block, 0, stream>>>                 //
            (in_data, out_errctrl, out_outlier, get_x(), get_y(), get_z(), get_leap().y, get_leap().z, radius, ebx2_r);
    }
    else {
        throw std::runtime_error("Lorenzo only works for 123-D.");
    }

    timer.timer_end(stream);
    if (stream)
        CHECK_CUDA(cudaStreamSynchronize(stream));
    else
        CHECK_CUDA(cudaDeviceSynchronize());

    time_elapsed = timer.get_time_elapsed();
}

template <typename T, typename E, typename FP>
template <bool DELAY_POSTQUANT>
void PredictorLorenzo<T, E, FP>::impl::reconstruct_proxy(
    T* const __restrict__ __in_outlier,
    T* const     in_anchor,
    E* const     in_errctrl,
    T* const     out_xdata,
    double const eb,
    int const    radius,
    cudaStream_t stream)
{
    // error bound
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;

    // decide if destructive for the input (outlier)
    auto in_outlier = __in_outlier == nullptr ? out_xdata : __in_outlier;

    cuda_timer_t timer;
    timer.timer_start(stream);

    if (get_ndim() == 1) {  // y-sequentiality == 8
        constexpr auto SEQ         = 8;
        constexpr auto DATA_SUBLEN = 256;
        auto           dim_block   = DATA_SUBLEN / SEQ;
        auto           dim_grid    = ConfigHelper::get_npart(get_x(), DATA_SUBLEN);

        ::cusz::x_lorenzo_1d1l<T, E, FP, DATA_SUBLEN, SEQ, DELAY_POSTQUANT>  //
            <<<dim_grid, dim_block, 0, stream>>>                             //
            (in_outlier, in_errctrl, out_xdata, get_x(), radius, ebx2);
    }
    else if (get_ndim() == 2) {  // y-sequentiality == 8
        auto dim_block = dim3(16, 2);
        auto dim_grid  = ConfigHelper::get_pardeg3(this->get_len3(), {16, 16, 1});

        ::cusz::x_lorenzo_2d1l_16x16data_mapto16x2<T, E, FP, DELAY_POSTQUANT>  //
            <<<dim_grid, dim_block, 0, stream>>>                               //
            (in_outlier, in_errctrl, out_xdata, get_x(), get_y(), get_leap().y, radius, ebx2);
    }
    else if (get_ndim() == 3) {  // y-sequentiality == 8
        auto dim_block = dim3(32, 1, 8);
        auto dim_grid  = ConfigHelper::get_pardeg3(this->get_len3(), {32, 8, 8});

        ::cusz::x_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, E, FP, DELAY_POSTQUANT>  //
            <<<dim_grid, dim_block, 0, stream>>>                                  //
            (in_outlier, in_errctrl, out_xdata, get_x(), get_y(), get_z(), get_leap().y, get_leap().z, radius, ebx2);
    }

    timer.timer_end(stream);
    if (stream)
        CHECK_CUDA(cudaStreamSynchronize(stream));
    else
        CHECK_CUDA(cudaDeviceSynchronize());

    time_elapsed = timer.get_time_elapsed();
}

}  // namespace cusz

#undef ALLOCDEV
#undef FREE_DEV_ARRAY
#undef DEFINE_ARRAY

#endif
