/**
 * @file prediction_impl.cuh
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
#include "../component/prediction.hh"
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

#include "../kernel/spline3.cuh"

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
#define THE_TYPE template <typename T, typename E, typename FP>
#define IMPL PredictionUnified<T, E, FP>::impl

namespace cusz {

THE_TYPE
IMPL::impl() {}

THE_TYPE
IMPL::~impl()
{
    FREE_DEV_ARRAY(anchor);
    FREE_DEV_ARRAY(errctrl);
}

THE_TYPE
void IMPL::clear_buffer() { cudaMemset(d_errctrl, 0x0, sizeof(E) * this->rtlen.assigned.quant); }

THE_TYPE
void IMPL::init(cusz_predictortype predictor, size_t x, size_t y, size_t z, bool dbg_print)
{
    auto len3 = dim3(x, y, z);
    init(predictor, len3, dbg_print);
}

THE_TYPE
void IMPL::init(cusz_predictortype predictor, dim3 xyz, bool dbg_print)
{
    this->derive_alloclen(predictor, xyz);

    // allocate
    ALLOCDEV2(anchor, T, this->alloclen.assigned.anchor);
    ALLOCDEV2(errctrl, E, this->alloclen.assigned.quant);
    if (not outlier_overlapped) ALLOCDEV(outlier, T, this->alloclen.assigned.outlier);

    if (dbg_print) this->debug_list_alloclen<T, E, FP>();
}

THE_TYPE
E* IMPL::expose_quant() const { return d_errctrl; }
THE_TYPE
E* IMPL::expose_errctrl() const { return d_errctrl; }
THE_TYPE
T* IMPL::expose_anchor() const { return d_anchor; }
THE_TYPE
T* IMPL::expose_outlier() const { return d_outlier; }

THE_TYPE
void IMPL::construct(
    cusz_predictortype predictor,
    dim3 const         len3,
    T*                 data_outlier,
    T**                anchor,
    E**                errctrl,
    double const       eb,
    int const          radius,
    cudaStream_t       stream)
{
    *anchor  = d_anchor;
    *errctrl = d_errctrl;

    if (predictor == LorenzoI) {
        derive_rtlen(LorenzoI, len3);
        this->check_rtlen();

        if (not delay_postquant)
            construct_proxy_LorenzoI<false>(data_outlier, d_anchor, d_errctrl, eb, radius, stream);
        else
            throw std::runtime_error("construct_proxy_LorenzoI<delay_postquant==true> not implemented.");
    }
    else if (predictor == Spline3) {
        this->derive_rtlen(Spline3, len3);
        this->check_rtlen();

        construct_proxy_Spline3(data_outlier, d_anchor, d_errctrl, eb, radius, stream);
    }
}

THE_TYPE
void IMPL::reconstruct(
    cusz_predictortype predictor,
    dim3               len3,
    T*                 outlier_xdata,
    T*                 anchor,
    E*                 errctrl,
    double const       eb,
    int const          radius,
    cudaStream_t       stream)
{
    if (predictor == LorenzoI) {
        this->derive_rtlen(LorenzoI, len3);
        this->check_rtlen();

        if (not delay_postquant)
            reconstruct_proxy_LorenzoI<false>(outlier_xdata, anchor, errctrl, eb, radius, stream);
        else
            throw std::runtime_error("construct_proxy_LorenzoI<delay_postquant==true> not implemented.");
    }
    else if (predictor == Spline3) {
        this->derive_rtlen(Spline3, len3);
        this->check_rtlen();
        // this->debug_list_rtlen<T, E, FP>(true);

        reconstruct_proxy_Spline3(outlier_xdata, anchor, errctrl, eb, radius, stream);
    }
}

THE_TYPE
float IMPL::get_time_elapsed() const { return time_elapsed; }

THE_TYPE
template <bool DELAY_POSTQUANT>
void IMPL::construct_proxy_LorenzoI(
    T* const     in_data,
    T* const     anchor,
    E* const     errctrl,
    double const eb,
    int const    radius,
    cudaStream_t stream)
{
    // error bound
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;

    auto out_outlier = in_data;

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
            (in_data, errctrl, out_outlier, get_x(), radius, ebx2_r);
    }
    else if (get_ndim() == 2) {  // y-sequentiality == 8
        auto dim_block = dim3(16, 2);
        auto dim_grid  = ConfigHelper::get_pardeg3(this->get_len3(), {16, 16, 1});

        ::cusz::c_lorenzo_2d1l_16x16data_mapto16x2<T, E, FP>  //
            <<<dim_grid, dim_block, 0, stream>>>              //
            (in_data, errctrl, out_outlier, get_x(), get_y(), get_leap().y, radius, ebx2_r);
    }
    else if (get_ndim() == 3) {  // y-sequentiality == 8
        auto dim_block = dim3(32, 1, 8);
        auto dim_grid  = ConfigHelper::get_pardeg3(this->get_len3(), {32, 8, 8});

        ::cusz::c_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, E, FP>  //
            <<<dim_grid, dim_block, 0, stream>>>                 //
            (in_data, errctrl, out_outlier, get_x(), get_y(), get_z(), get_leap().y, get_leap().z, radius, ebx2_r);
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

THE_TYPE
template <bool DELAY_POSTQUANT>
void IMPL::reconstruct_proxy_LorenzoI(
    T*           out_xdata,
    T*           anchor,
    E*           errctrl,
    double const eb,
    int const    radius,
    cudaStream_t stream)
{
    // error bound
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;

    // decide if destructive for the input (outlier)
    // auto in_outlier = __in_outlier == nullptr ? out_xdata : __in_outlier;
    auto in_outlier = out_xdata;

    cuda_timer_t timer;
    timer.timer_start(stream);

    if (get_ndim() == 1) {  // y-sequentiality == 8
        constexpr auto SEQ         = 8;
        constexpr auto DATA_SUBLEN = 256;
        auto           dim_block   = DATA_SUBLEN / SEQ;
        auto           dim_grid    = ConfigHelper::get_npart(get_x(), DATA_SUBLEN);

        ::cusz::x_lorenzo_1d1l<T, E, FP, DATA_SUBLEN, SEQ, DELAY_POSTQUANT>  //
            <<<dim_grid, dim_block, 0, stream>>>                             //
            (in_outlier, errctrl, out_xdata, get_x(), radius, ebx2);
    }
    else if (get_ndim() == 2) {  // y-sequentiality == 8
        auto dim_block = dim3(16, 2);
        auto dim_grid  = ConfigHelper::get_pardeg3(this->get_len3(), {16, 16, 1});

        ::cusz::x_lorenzo_2d1l_16x16data_mapto16x2<T, E, FP, DELAY_POSTQUANT>  //
            <<<dim_grid, dim_block, 0, stream>>>                               //
            (in_outlier, errctrl, out_xdata, get_x(), get_y(), get_leap().y, radius, ebx2);
    }
    else if (get_ndim() == 3) {  // y-sequentiality == 8
        auto dim_block = dim3(32, 1, 8);
        auto dim_grid  = ConfigHelper::get_pardeg3(this->get_len3(), {32, 8, 8});

        ::cusz::x_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, E, FP, DELAY_POSTQUANT>  //
            <<<dim_grid, dim_block, 0, stream>>>                                  //
            (in_outlier, errctrl, out_xdata, get_x(), get_y(), get_z(), get_leap().y, get_leap().z, radius, ebx2);
    }

    timer.timer_end(stream);
    if (stream)
        CHECK_CUDA(cudaStreamSynchronize(stream));
    else
        CHECK_CUDA(cudaDeviceSynchronize());

    time_elapsed = timer.get_time_elapsed();
}

THE_TYPE
void IMPL::impl::construct_proxy_Spline3(
    T*           in_data,
    T*&          anchor,
    E*&          errctrl,
    double const eb,
    int const    radius,
    cudaStream_t stream)
{
    auto ebx2 = eb * 2;
    auto eb_r = 1 / eb;

    anchor  = d_anchor;
    errctrl = d_errctrl;

    cuda_timer_t timer;
    timer.timer_start();

    cusz::c_spline3d_infprecis_32x8x8data<T*, E*, float, 256, false>
        <<<this->rtlen.nblock, dim3(256, 1, 1), 0, stream>>>             //
        (in_data, this->rtlen.base.len3, this->rtlen.base.leap,          //
         d_errctrl, this->rtlen.aligned.len3, this->rtlen.aligned.leap,  //
         d_anchor, this->rtlen.anchor.leap,                              //
         eb_r, ebx2, radius);

    timer.timer_end();

    if (stream)
        CHECK_CUDA(cudaStreamSynchronize(stream));
    else
        CHECK_CUDA(cudaDeviceSynchronize());

    this->time_elapsed = timer.get_time_elapsed();
}

THE_TYPE
void IMPL::impl::reconstruct_proxy_Spline3(
    T*           xdata,
    T*           anchor,
    E*           errctrl,
    double const eb,
    int const    radius,
    cudaStream_t stream)
{
    auto ebx2 = eb * 2;
    auto eb_r = 1 / eb;

    cuda_timer_t timer;
    timer.timer_start();

    cusz::x_spline3d_infprecis_32x8x8data<E*, T*, float, 256><<<this->rtlen.nblock, dim3(256, 1, 1), 0, stream>>>  //
        (errctrl, this->rtlen.aligned.len3, this->rtlen.aligned.leap,                                              //
         anchor, this->rtlen.anchor.len3, this->rtlen.anchor.leap,                                                 //
         xdata, this->rtlen.base.len3, this->rtlen.base.leap,                                                      //
         eb_r, ebx2, radius);

    timer.timer_end();

    if (stream)
        CHECK_CUDA(cudaStreamSynchronize(stream));
    else
        CHECK_CUDA(cudaDeviceSynchronize());

    this->time_elapsed = timer.get_time_elapsed();
}

}  // namespace cusz

#undef ALLOCDEV
#undef FREE_DEV_ARRAY
#undef DEFINE_ARRAY

#undef THE_TYPE
#undef IMPL

#endif
