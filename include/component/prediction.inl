/**
 * @file prediction.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-23
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef FB315D3E_6B96_4F5D_9975_F35702205BC1
#define FB315D3E_6B96_4F5D_9975_F35702205BC1

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include "../common.hh"
#include "../kernel/cpplaunch_cuda.hh"
#include "../kernel/lorenzo_all.hh"
#include "../utils.hh"

#include "cusz/type.h"
#include "pred_boilerplate_deprecated.hh"

#define DEFINE_ARRAY(VAR, TYPE) TYPE* d_##VAR{nullptr};

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

namespace cusz {

template <typename T, typename E, typename FP>
class PredictionUnified : public PredictorBoilerplate {
   public:
    using Origin    = T;
    using Anchor    = T;
    using ErrCtrl   = E;
    using Precision = FP;

   public:
    ~PredictionUnified()
    {  // dtor
        FREE_DEV_ARRAY(anchor);
        FREE_DEV_ARRAY(errctrl);
        FREE_DEV_ARRAY(outlier);
    }
    PredictionUnified() {}                                   // ctor
    PredictionUnified(const PredictionUnified&);             // copy ctor
    PredictionUnified& operator=(const PredictionUnified&);  // copy assign
    PredictionUnified(PredictionUnified&&);                  // move ctor
    PredictionUnified& operator=(PredictionUnified&&);       // move assign

    void init(cusz_predictortype predictor, size_t x, size_t y, size_t z, bool dbg_print = false)
    {
        auto len3 = dim3(x, y, z);
        init(predictor, len3, dbg_print);
    }
    void init(cusz_predictortype predictor, dim3 xyz, bool dbg_print = false)
    {
        this->derive_alloclen(predictor, xyz);

        // allocate
        ALLOCDEV2(anchor, T, this->alloclen.assigned.anchor);
        ALLOCDEV2(errctrl, E, this->alloclen.assigned.quant);
        ALLOCDEV2(outlier, T, this->alloclen.assigned.outlier);

        if (dbg_print) this->debug_list_alloclen<T, E, FP>();
    }

    void construct(
        cusz_predictortype predictor,
        dim3 const         len3,
        T*                 data,
        T**                ptr_anchor,
        E**                ptr_errctrl,
        T**                ptr_outlier,
        double const       eb,
        int const          radius,
        cudaStream_t       stream)
    {
        *ptr_anchor  = d_anchor;
        *ptr_errctrl = d_errctrl;
        *ptr_outlier = d_outlier;

        if (predictor == LorenzoI) {
            derive_rtlen(LorenzoI, len3);
            this->check_rtlen();

            // ad hoc placeholder
            // auto      anchor_len3  = dim3(0, 0, 0);
            // auto      errctrl_len3 = dim3(0, 0, 0);
            uint32_t* outlier_idx = nullptr;

            compress_predict_lorenzo_i<T, E, FP>(
                data, len3, eb, radius,                      //
                d_errctrl, d_outlier, outlier_idx, nullptr,  //
                &time_elapsed, stream);
        }
        else if (predictor == Spline3) {
            throw std::runtime_error("spline3 is disabled in this version.");
            // this->derive_rtlen(Spline3, len3);
            // this->check_rtlen();
            // cusz::cpplaunch_construct_Spline3<T, E, FP>(
            //     true,  //
            //     data, len3, d_anchor, this->rtlen.anchor.len3, d_errctrl, this->rtlen.aligned.len3, eb, radius,
            //     &time_elapsed, stream);
        }
    }

    void reconstruct(
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

            // ad hoc placeholder
            // auto      anchor_len3  = dim3(0, 0, 0);
            // auto      errctrl_len3 = dim3(0, 0, 0);
            auto      xdata       = outlier_xdata;
            auto      outlier     = outlier_xdata;
            uint32_t* outlier_idx = nullptr;

            auto xdata_len3 = len3;

            decompress_predict_lorenzo_i<T, E, FP>(
                errctrl, xdata_len3, outlier, outlier_idx, 0, eb, radius,  //
                xdata,                                                     //
                &time_elapsed, stream);
        }
        else if (predictor == Spline3) {
            throw std::runtime_error("spline3 is disabled in this version.");
            // this->derive_rtlen(Spline3, len3);
            // this->check_rtlen();
            // cusz::cpplaunch_reconstruct_Spline3<T, E, FP>(
            //     outlier_xdata, len3, anchor, this->rtlen.anchor.len3, errctrl, this->rtlen.aligned.len3, eb, radius,
            //     &time_elapsed, stream);
        }
    }

    void clear_buffer() { cudaMemset(d_errctrl, 0x0, sizeof(E) * this->rtlen.assigned.quant); }

    float get_time_elapsed() const { return time_elapsed; }
    // size_t get_alloclen_data() const;
    // size_t get_alloclen_quant() const;
    // size_t get_len_data() const;
    // size_t get_len_quant() const;
    // size_t get_len_anchor() const;

    E* expose_quant() const { return d_errctrl; }
    E* expose_errctrl() const { return d_errctrl; }
    T* expose_anchor() const { return d_anchor; }
    T* expose_outlier() const { return d_outlier; }

   public:
    // data
    DEFINE_ARRAY(anchor, T);
    DEFINE_ARRAY(errctrl, E);
    DEFINE_ARRAY(outlier, T);
};

}  // namespace cusz

#undef ALLOCDEV
#undef FREE_DEV_ARRAY
#undef DEFINE_ARRAY

#endif /* FB315D3E_6B96_4F5D_9975_F35702205BC1 */
