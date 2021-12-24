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
    PredictorLorenzo(dim3 xyz, bool delay_postquant);
    ~PredictorLorenzo(){};

    // helper
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
};

}  // namespace cusz

#endif