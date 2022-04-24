/**
 * @file predictors.hh
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
#include "predictor_boilerplate.hh"

#define DEFINE_ARRAY(VAR, TYPE) TYPE* d_##VAR{nullptr};

namespace cusz {

template <typename T, typename E, typename FP>
class PredictorInterface {
   public:
    virtual void init(size_t, size_t, size_t, bool)                                         = 0;
    virtual void init(dim3, bool)                                                           = 0;
    virtual void construct(dim3 const, T*, T*&, E*&, double const, int const, cudaStream_t) = 0;
    virtual void reconstruct(dim3, T*&, T*, E*, double const, int const, cudaStream_t)      = 0;

    // TOOD too scattered
    virtual E* expose_quant() const   = 0;
    virtual E* expose_errctrl() const = 0;
    virtual T* expose_anchor() const  = 0;
    virtual T* expose_outlier() const = 0;

    virtual float get_time_elapsed() const = 0;

    virtual void clear_buffer() = 0;
};

namespace api {

template <typename T, typename E, typename FP>
class PredictorLorenzo : public PredictorInterface<T, E, FP> {
   public:
    using Origin    = T;
    using Anchor    = T;
    using ErrCtrl   = E;
    using Precision = FP;

   public:
    // TODO will change to private
    struct impl;

    void init(size_t, size_t, size_t, bool dbg_print = false);
    void init(dim3, bool = false);
    void construct(dim3 const, T*, T*&, E*&, double const, int const, cudaStream_t = nullptr);
    void reconstruct(dim3, T*&, T*, E*, double const, int const, cudaStream_t = nullptr);

    void expose_internal(E*&, T*&, T*&);

    // TOOD too scattered
    E* expose_quant() const;
    E* expose_errctrl() const;
    T* expose_anchor() const;
    T* expose_outlier() const;

    float get_time_elapsed() const;

    void clear_buffer();

   private:
};

template <typename T, typename E, typename FP>
struct PredictorLorenzo<T, E, FP>::impl : public PredictorBoilerplate {
    // TODO remove the placeholder below
   public:
    using Origin    = T;
    using Anchor    = T;
    using ErrCtrl   = E;
    using Precision = FP;

   public:
    impl();
    ~impl();

    void init(size_t, size_t, size_t, bool = false);
    void init(dim3, bool = false);
    void construct(dim3 const, T*, T*&, E*&, double const, int const, cudaStream_t = nullptr);
    void construct(dim3 const, T*, T*&, E*&, T*&, double const, int const, cudaStream_t = nullptr);
    void reconstruct(dim3, T*&, T*, E*, double const, int const, cudaStream_t = nullptr);
    void reconstruct(dim3, T*, T*, E*, T*&, double const, int const, cudaStream_t = nullptr);

    void expose_internal(E*&, T*&, T*&);

    // TOOD too scattered
    E* expose_quant() const;
    E* expose_errctrl() const;
    T* expose_anchor() const;
    T* expose_outlier() const;

    void clear_buffer();

   private:
    // data
    DEFINE_ARRAY(anchor, T);
    DEFINE_ARRAY(errctrl, E);
    DEFINE_ARRAY(outlier, T);
    // flags
    bool delay_postquant{false};
    bool outlier_overlapped{true};

    template <bool DELAY_POSTQUANT>
    void construct_proxy(T* const, T* const, E* const, T* const, double const, int const, cudaStream_t const = nullptr);

    template <bool DELAY_POSTQUANT>
    void
    reconstruct_proxy(T* const, T* const, E* const, T* const, double const, int const, cudaStream_t const = nullptr);
};

template <typename T, typename E, typename FP>
class PredictorSpline3 : public PredictorInterface<T, E, FP> {
   public:
    using Origin    = T;
    using Anchor    = T;
    using ErrCtrl   = E;
    using Precision = FP;

   public:
    // TODO will change to private
    struct impl;

    void init(size_t, size_t, size_t, bool dbg_print = false);
    void init(dim3, bool = false);
    void construct(dim3 const, T*, T*&, E*&, double const, int const, cudaStream_t = nullptr);
    void reconstruct(dim3, T*&, T*, E*, double const, int const, cudaStream_t = nullptr);

    void expose_internal(E*&, T*&, T*&);

    // TOOD too scattered
    E* expose_quant() const;
    E* expose_errctrl() const;
    T* expose_anchor() const;
    T* expose_outlier() const;

    void clear_buffer();

    float get_time_elapsed() const;

   private:
};

template <typename T, typename E, typename FP>
struct PredictorSpline3<T, E, FP>::impl : public PredictorBoilerplate {
    // TODO remove the placeholder below
   public:
    using Origin    = T;
    using Anchor    = T;
    using ErrCtrl   = E;
    using Precision = FP;

   public:
    impl();
    ~impl();

    void init(size_t, size_t, size_t, bool = false);
    void init(dim3, bool = false);
    void construct(dim3 const, T*, T*&, E*&, double const, int const, cudaStream_t = nullptr);
    void construct(dim3 const, T*, T*&, E*&, T*&, double const, int const, cudaStream_t = nullptr);
    void reconstruct(dim3, T*&, T*, E*, double const, int const, cudaStream_t = nullptr);
    void reconstruct(dim3, T*, T*, E*, T*&, double const, int const, cudaStream_t = nullptr);

    void expose_internal(E*&, T*&, T*&);

    // TOOD too scattered
    E* expose_quant() const;
    E* expose_errctrl() const;
    T* expose_anchor() const;
    T* expose_outlier() const;

    void clear_buffer();

   private:
    // data
    DEFINE_ARRAY(anchor, T);
    DEFINE_ARRAY(errctrl, E);
    DEFINE_ARRAY(outlier, T);
    // flags
    bool delay_postquant{false};
    bool outlier_overlapped{true};

    // unique to spline3
   private:
    static const auto BLOCK = 8;

    using TITER = T*;
    using EITER = E*;

   private:
    bool dbg_mode{false};

    bool delay_postquant_dummy;

   private:
    void init_continue(bool _delay_postquant_dummy = false, bool _outlier_overlapped = true);
    // override
    void   derive_alloclen(dim3);
    void   derive_rtlen(dim3);
    size_t get_alloclen_quant() const;
    size_t get_len_quant() const;
};

}  // namespace api
}  // namespace cusz

#undef DEFINE_ARRAY

#endif
