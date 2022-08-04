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
#include "../kernel/launch_prediction.cuh"
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
        CHECK_CUDA(hipMalloc(&d_##VAR, NBYTE));     \
        CHECK_CUDA(hipMemset(d_##VAR, 0x0, NBYTE)); \
    }

#define ALLOCDEV2(VAR, TYPE, LEN)                                 \
    if (LEN != 0) {                                               \
        CHECK_CUDA(hipMalloc(&d_##VAR, sizeof(TYPE) * LEN));     \
        CHECK_CUDA(hipMemset(d_##VAR, 0x0, sizeof(TYPE) * LEN)); \
    }

#define FREE_DEV_ARRAY(VAR)            \
    if (d_##VAR) {                     \
        CHECK_CUDA(hipFree(d_##VAR)); \
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
void IMPL::clear_buffer() { hipMemset(d_errctrl, 0x0, sizeof(E) * this->rtlen.assigned.quant); }

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
    hipStream_t       stream)
{
    *anchor  = d_anchor;
    *errctrl = d_errctrl;

    if (predictor == LorenzoI) {
        derive_rtlen(LorenzoI, len3);
        this->check_rtlen();

        auto placeholder1 = dim3(0, 0, 0);
        auto placeholder2 = dim3(0, 0, 0);

        if (not delay_postquant)
            launch_construct_LorenzoI<T, E, FP, false>(
                data_outlier, len3, d_anchor, placeholder1, d_errctrl, placeholder2, eb, radius, time_elapsed, stream);
        else
            launch_construct_LorenzoI<T, E, FP, true>(
                data_outlier, len3, d_anchor, placeholder1, d_errctrl, placeholder2, eb, radius, time_elapsed, stream);
    }
    else if (predictor == Spline3) {
        this->derive_rtlen(Spline3, len3);
        this->check_rtlen();

        launch_construct_Spline3<T, E, FP, true>(
            data_outlier, len3, d_anchor, this->rtlen.anchor.len3, d_errctrl, this->rtlen.aligned.len3, eb, radius,
            time_elapsed, stream);
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
    hipStream_t       stream)
{
    if (predictor == LorenzoI) {
        this->derive_rtlen(LorenzoI, len3);
        this->check_rtlen();

        auto placeholder1 = dim3(0, 0, 0);
        auto placeholder2 = dim3(0, 0, 0);

        launch_reconstruct_LorenzoI<T, E, FP>(
            outlier_xdata, len3, anchor, placeholder1, errctrl, placeholder2, eb, radius, time_elapsed, stream);
    }
    else if (predictor == Spline3) {
        this->derive_rtlen(Spline3, len3);
        this->check_rtlen();
        // this->debug_list_rtlen<T, E, FP>(true);

        launch_reconstruct_Spline3<T, E, FP>(
            outlier_xdata, len3, anchor, this->rtlen.anchor.len3, errctrl, this->rtlen.aligned.len3, eb, radius,
            time_elapsed, stream);
    }
}

THE_TYPE
float IMPL::get_time_elapsed() const { return time_elapsed; }

}  // namespace cusz

#undef ALLOCDEV
#undef FREE_DEV_ARRAY
#undef DEFINE_ARRAY

#undef THE_TYPE
#undef IMPL

#endif
