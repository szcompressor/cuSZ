/**
 * @file extrap_lorenzo.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-06-16
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_WRAPPER_EXTRAP_LORENZO_CUH
#define CUSZ_WRAPPER_EXTRAP_LORENZO_CUH

#include "../../include/predictor.hh"

namespace cusz {

template <typename T, typename E, typename FP>
class PredictorLorenzo : public PredictorAbstraction<T, E> {
   public:
    using Precision = FP;

   private:
    dim3     size;  // size.x, size.y, size.z
    dim3     leap;  // leap.y, leap.z
    int      ndim;
    uint32_t len_data;
    uint32_t len_outlier;
    uint32_t len_quant;  // may differ from `len_data`
    bool     delay_postquant;
    bool     outlier_overlapped;

    float time_elapsed;

    struct {
        bool count_nnz;
        // bool blockwide_gather; // future use
    } on_off;

    template <bool DELAY_POSTQUANT>
    void construct_proxy(
        T* in_data,
        T* out_anchor,
        E* out_errctrl,
        T* __restrict__ __out_outlier,
        double const eb,
        int const    radius,
        cudaStream_t const = nullptr);

    template <bool DELAY_POSTQUANT>
    void reconstruct_proxy(
        T* __restrict__ __in_outlier,
        T*           in_anchor,
        E*           in_errctrl,
        T*           out_xdata,
        double const eb,
        int const    radius,
        cudaStream_t const = nullptr);

   public:
    // constructor
    PredictorLorenzo() = default;
    PredictorLorenzo(dim3 xyz, bool delay_postquant);
    ~PredictorLorenzo()
    {
#define LORENZO_FREEDEV(VAR) \
    if (d_##VAR) {           \
        cudaFree(d_##VAR);   \
        d_##VAR = nullptr;   \
    }
        LORENZO_FREEDEV(anchor);
        LORENZO_FREEDEV(errctrl);

#undef LORENZO_FREEDEV
    };

    // helper
    uint32_t get_data_len() const { return len_data; }
    uint32_t get_anchor_len() const { return 0; }
    uint32_t get_quant_len() const { return len_quant; }
    uint32_t get_outlier_len() const { return len_outlier; }
    uint32_t get_workspace_nbyte() const { return 0; };

    float get_time_elapsed() const { return time_elapsed; }

    // methods
    void dryrun(T* in_out);

    void construct(
        T*           in_data,
        T*           out_anchor,
        E*           out_errctrl,
        double const eb,
        int const    radius,
        cudaStream_t const = nullptr,
        T* __restrict__    = nullptr);

    void reconstruct(
        T*           in_anchor,
        E*           in_errctrl,
        T*           out_xdata,
        double const eb,
        int const    radius,
        cudaStream_t const = nullptr,
        T* __restrict__    = nullptr);

    // refactor below

   private:
#define DEFINE_ARRAY(VAR, TYPE) TYPE* d_##VAR{nullptr};
    DEFINE_ARRAY(anchor, T);
    DEFINE_ARRAY(errctrl, E);
    DEFINE_ARRAY(outlier, T);
#undef DEFINE_ARRAY

   public:
    void allocate_workspace(dim3 _size3, bool _delay_postquant = false, bool _outlier_overlapped = true)
    {
        // size
        size      = _size3;
        leap      = dim3(1, size.x, size.x * size.y);
        len_data  = size.x * size.y * size.z;
        len_quant = len_data;

        len_outlier = len_data;

        ndim = 3;
        if (size.z == 1) ndim = 2;
        if (size.z == 1 and size.y == 1) ndim = 1;

        // on off
        delay_postquant    = _delay_postquant;
        outlier_overlapped = _outlier_overlapped;

        // allocate
#define ALLOCDEV(VAR, SYM, NBYTE) \
    cudaMalloc(&d_##VAR, NBYTE);  \
    cudaMemset(d_##VAR, 0x0, NBYTE);
        ALLOCDEV(anchor, T, 0);  // for lorenzo, anchor can be 0
        ALLOCDEV(errctrl, E, sizeof(E) * len_quant);
        if (not outlier_overlapped) ALLOCDEV(outlier, T, sizeof(T) * len_data);
#undef ALLCDEV
    }

   public:
    // E* expose_errctrl() const { return d_errctrl; }
    // T* expose_anchor() const { return d_anchor; }
    // T* expose_outlier() const { return d_outlier; }

    void construct(
        T* __restrict__ in_data,
        double const eb,
        int const    radius,
        T*&          out_anchor,
        E*&          out_errctrl,
        T*& __restrict__ out_outlier,
        cudaStream_t const stream = nullptr)
    {
        out_anchor  = d_anchor;
        out_errctrl = d_errctrl;
        out_outlier = d_outlier;

        if (not delay_postquant)
            construct_proxy<false>(in_data, out_anchor, out_errctrl, out_outlier, eb, radius, stream);
        else
            throw std::runtime_error("construct_proxy<delay_postquant==true> not implemented.");
    }

    void construct(
        T*                 in_data__out_outlier,
        double const       eb,
        int const          radius,
        T*&                out_anchor,
        E*&                out_errctrl,
        cudaStream_t const stream = nullptr)
    {
        out_anchor  = d_anchor;
        out_errctrl = d_errctrl;

        if (not delay_postquant)
            construct_proxy<false>(in_data__out_outlier, out_anchor, out_errctrl, nullptr, eb, radius, stream);
        else
            throw std::runtime_error("construct_proxy<delay_postquant==true> not implemented.");
    }

    void reconstruct(
        T* __restrict__ in_outlier,
        T*           in_anchor,
        E*           in_errctrl,
        double const eb,
        int const    radius,
        T*& __restrict__ out_xdata,
        cudaStream_t const stream = nullptr)
    {
        if (not delay_postquant)
            reconstruct_proxy<false>(in_outlier, in_anchor, in_errctrl, out_xdata, eb, radius, stream);
        else
            throw std::runtime_error("construct_proxy<delay_postquant==true> not implemented.");
    }

    void reconstruct(
        T*                 in_anchor,
        E*                 in_errctrl,
        double const       eb,
        int const          radius,
        T*&                in_outlier__out_xdata,
        cudaStream_t const stream = nullptr)
    {
        if (not delay_postquant)
            reconstruct_proxy<false>(nullptr, in_anchor, in_errctrl, in_outlier__out_xdata, eb, radius, stream);
        else
            throw std::runtime_error("construct_proxy<delay_postquant==true> not implemented.");
    }
};

}  // namespace cusz

#endif