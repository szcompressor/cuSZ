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

#ifndef CUSZ_COMPONENT_PREDICTORS_HH
#define CUSZ_COMPONENT_PREDICTORS_HH

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>

#include "cusz/type.h"
#include "predictor_boilerplate.hh"

#define DEFINE_ARRAY(VAR, TYPE) TYPE* d_##VAR{nullptr};

namespace cusz {

template <typename T, typename E, typename FP>
class PredictionUnified {
   public:
    using Origin    = T;
    using Anchor    = T;
    using ErrCtrl   = E;
    using Precision = FP;

   private:
    class impl;
    std::unique_ptr<impl> pimpl;

   public:
    ~PredictionUnified();                                    // dtor
    PredictionUnified();                                     // ctor
    PredictionUnified(const PredictionUnified&);             // copy ctor
    PredictionUnified& operator=(const PredictionUnified&);  // copy assign
    PredictionUnified(PredictionUnified&&);                  // move ctor
    PredictionUnified& operator=(PredictionUnified&&);       // move assign

    void init(cusz_predictortype, size_t, size_t, size_t, bool dbg_print = false);
    void init(cusz_predictortype, dim3, bool = false);

    void construct(
        cusz_predictortype predictor,
        dim3 const         len3,
        T*                 data_outlier,
        T**                anchor,
        E**                errctrl,
        double const       eb,
        int const          radius,
        cudaStream_t       stream);

    void reconstruct(
        cusz_predictortype predictor,
        dim3               len3,
        T*                 outlier_xdata,
        T*                 anchor,
        E*                 errctrl,
        double const       eb,
        int const          radius,
        cudaStream_t       stream);

    void clear_buffer();

    float  get_time_elapsed() const;
    size_t get_alloclen_data() const;
    size_t get_alloclen_quant() const;
    size_t get_len_data() const;
    size_t get_len_quant() const;
    size_t get_len_anchor() const;
    // TODO void expose_internal(E*&, T*&, T*&);
    E* expose_quant() const;
    E* expose_errctrl() const;
    T* expose_anchor() const;
    T* expose_outlier() const;
};

template <typename T, typename E, typename FP>
class PredictionUnified<T, E, FP>::impl : public PredictorBoilerplate {
    // TODO remove the placeholder below
   public:
    using Origin    = T;
    using Anchor    = T;
    using ErrCtrl   = E;
    using Precision = FP;

   public:
    impl();
    ~impl();

    void init(cusz_predictortype, size_t, size_t, size_t, bool = false);
    void init(cusz_predictortype, dim3, bool = false);

    void construct(
        cusz_predictortype predictor,
        dim3 const         len3,
        T*                 data_outlier,
        T**                anchor,
        E**                errctrl,
        double const       eb,
        int const          radius,
        cudaStream_t       stream);

    void reconstruct(
        cusz_predictortype predictor,
        dim3               len3,
        T*                 outlier_xdata,
        T*                 anchor,
        E*                 errctrl,
        double const       eb,
        int const          radius,
        cudaStream_t       stream);

    void clear_buffer();

    float get_time_elapsed() const;
    // TODO void expose_internal(E*&, T*&, T*&);
    E* expose_quant() const;
    E* expose_errctrl() const;
    T* expose_anchor() const;
    T* expose_outlier() const;

   private:
    // data
    DEFINE_ARRAY(anchor, T);
    DEFINE_ARRAY(errctrl, E);
    DEFINE_ARRAY(outlier, T);
    // flags
    // bool delay_postquant{false};
    bool outlier_overlapped{true};

    template <bool NO_R_SEPARATE>
    void construct_proxy_LorenzoI(T*, T*, E*, double const, int const, cudaStream_t = nullptr);

    void reconstruct_proxy_LorenzoI(T*, T*, E*, double const, int const, cudaStream_t = nullptr);

    void construct_proxy_Spline3(
        T*           data_outlier,
        T*&          anchor,
        E*&          errctrl,
        double const eb,
        int const    radius,
        cudaStream_t stream);

    void reconstruct_proxy_Spline3(
        T*           outlier_xdata,
        T*           anchor,
        E*           errctrl,
        double const eb,
        int const    radius,
        cudaStream_t stream);
};

// template <typename T, typename E, typename FP>
// class PredictorSpline3 {
//    public:
//     using Origin    = T;
//     using Anchor    = T;
//     using ErrCtrl   = E;
//     using Precision = FP;

//    private:
//     class impl;
//     std::unique_ptr<impl> pimpl;

//    public:
//     ~PredictorSpline3();                                   // dtor
//     PredictorSpline3();                                    // ctor
//     PredictorSpline3(const PredictorSpline3&);             // copy ctor
//     PredictorSpline3& operator=(const PredictorSpline3&);  // copy assign
//     PredictorSpline3(PredictorSpline3&&);                  // move ctor
//     PredictorSpline3& operator=(PredictorSpline3&&);       // move assign

//     void init(size_t, size_t, size_t, bool dbg_print = false);
//     void init(dim3, bool = false);
//     void construct(dim3 const, T*, T*&, E*&, double const, int const, cudaStream_t = nullptr);
//     void reconstruct(dim3, T*&, T*, E*, double const, int const, cudaStream_t = nullptr);
//     void clear_buffer();

//     float  get_time_elapsed() const;
//     size_t get_alloclen_data() const;
//     size_t get_alloclen_quant() const;
//     size_t get_len_data() const;
//     size_t get_len_quant() const;
//     size_t get_len_anchor() const;
//     // TODO void expose_internal(E*&, T*&, T*&);
//     E* expose_quant() const;
//     E* expose_errctrl() const;
//     T* expose_anchor() const;
//     T* expose_outlier() const;
// };

// template <typename T, typename E, typename FP>
// class PredictorSpline3<T, E, FP>::impl : public PredictorBoilerplate {
//     // TODO remove the placeholder below
//    public:
//     using Origin    = T;
//     using Anchor    = T;
//     using ErrCtrl   = E;
//     using Precision = FP;

//    public:
//     impl();
//     ~impl();

//     void init(size_t, size_t, size_t, bool = false);
//     void init(dim3, bool = false);
//     void construct(dim3 const, T*, T*&, E*&, double const, int const, cudaStream_t = nullptr);
//     void construct(dim3 const, T*, T*&, E*&, T*&, double const, int const, cudaStream_t = nullptr);
//     void reconstruct(dim3, T*&, T*, E*, double const, int const, cudaStream_t = nullptr);
//     void reconstruct(dim3, T*, T*, E*, T*&, double const, int const, cudaStream_t = nullptr);
//     void clear_buffer();

//     float get_time_elapsed() const;
//     // TODO void expose_internal(E*&, T*&, T*&);
//     E* expose_quant() const;
//     E* expose_errctrl() const;
//     T* expose_anchor() const;
//     T* expose_outlier() const;
//     // override for spline3
//     // size_t get_alloclen_quant() const;
//     // size_t get_len_quant() const;

//    private:
//     // data
//     DEFINE_ARRAY(anchor, T);
//     DEFINE_ARRAY(errctrl, E);
//     DEFINE_ARRAY(outlier, T);
//     // flags
//     bool delay_postquant{false};
//     bool outlier_overlapped{true};

//     // unique to spline3
//    private:
//     static const auto BLOCK = 8;

//     using TITER = T*;
//     using EITER = E*;

//    private:
//     bool dbg_mode{false};
//     bool delay_postquant_dummy;

//    private:
//     void init_continue(bool _delay_postquant_dummy = false, bool _outlier_overlapped = true);
//     // override
//     void derive_alloclen(dim3, cusz_predictortype = LorenzoI);
//     void derive_rtlen(dim3, cusz_predictortype = LorenzoI);
// };

}  // namespace cusz

#undef DEFINE_ARRAY

#endif
