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
#include "../kernel/lorenzo_all.hh"
#include "../utils.hh"
#include "cusz/type.h"

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
class Predictor {
   public:
    using Origin    = T;
    using Anchor    = T;
    using ErrCtrl   = E;
    using Precision = FP;

   private:
    float time_elapsed;

    struct DerivedLengths {
        struct {
            dim3   len3, leap;
            size_t serialized;

            void set_leap() { leap = ConfigHelper::get_leap(len3); }
            void set_serialized() { serialized = ConfigHelper::get_serialized_len(len3); }
        } base, anchor, aligned;

        dim3 nblock;
        int  ndim;

        struct {
            size_t data, quant, outlier, anchor;
        } assigned;

        // dim3 get_len3() const { return base.len3; }
        dim3 get_leap() const { return base.leap; }
    } rtlen;

    template <class DERIVED>
    void __derive_len(dim3 base, DERIVED& derived)
    {
        derived.base.len3 = base;
        derived.base.set_leap();
        derived.base.set_serialized();
        derived.ndim = ConfigHelper::get_ndim(base);

        derived.assigned.data    = derived.base.serialized;
        derived.assigned.quant   = derived.base.serialized;
        derived.assigned.outlier = derived.base.serialized;
        derived.assigned.anchor  = 0;
    }

   public:
    ~Predictor()
    {  // dtor
        FREE_DEV_ARRAY(anchor);
        FREE_DEV_ARRAY(errctrl);
        FREE_DEV_ARRAY(outlier);
    }
    Predictor() {}                           // ctor
    Predictor(const Predictor&);             // copy ctor
    Predictor& operator=(const Predictor&);  // copy assign
    Predictor(Predictor&&);                  // move ctor
    Predictor& operator=(Predictor&&);       // move assign

    size_t get_len_data() const { return rtlen.assigned.data; }
    size_t get_len_anchor() const { return rtlen.assigned.anchor; }
    size_t get_len_quant() const { return rtlen.assigned.quant; }
    size_t get_len_outlier() const { return rtlen.assigned.outlier; }
    dim3   get_leap() const { return this->rtlen.get_leap(); }
    int    get_ndim() const { return this->rtlen.ndim; }

    void derive_lengths(cusz_predictortype predictor, dim3 base) { this->__derive_len(base, this->rtlen); }

    void init(cusz_predictortype predictor, size_t x, size_t y, size_t z, bool dbg_print = false)
    {
        auto len3 = dim3(x, y, z);
        init(predictor, len3, dbg_print);
    }
    void init(cusz_predictortype predictor, dim3 xyz, bool dbg_print = false)
    {
        this->derive_lengths(predictor, xyz);

        // allocate
        ALLOCDEV2(anchor, T, this->rtlen.assigned.anchor);
        ALLOCDEV2(errctrl, E, this->rtlen.assigned.quant);
        ALLOCDEV2(outlier, T, this->rtlen.assigned.outlier);
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
        // derive_lengths(LorenzoI, len3);

        *ptr_anchor  = d_anchor;
        *ptr_errctrl = d_errctrl;
        *ptr_outlier = d_outlier;

        uint32_t* outlier_idx = nullptr;

        compress_predict_lorenzo_i<T, E, FP>(
            data, len3, eb, radius,                      //
            d_errctrl, d_outlier, outlier_idx, nullptr,  //
            &time_elapsed, stream);
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
        // derive_lengths(LorenzoI, len3);

        auto      xdata       = outlier_xdata;
        auto      outlier     = outlier_xdata;
        uint32_t* outlier_idx = nullptr;

        auto xdata_len3 = len3;

        decompress_predict_lorenzo_i<T, E, FP>(
            errctrl, xdata_len3, outlier, outlier_idx, 0, eb, radius,  //
            xdata,                                                     //
            &time_elapsed, stream);
    }

    void clear_buffer() { cudaMemset(d_errctrl, 0x0, sizeof(E) * this->rtlen.assigned.quant); }

    float get_time_elapsed() const { return time_elapsed; }

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
