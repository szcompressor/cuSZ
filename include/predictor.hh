/**
 * @file predictor.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-15
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_INCLUDE_PREDICTOR_HH
#define CUSZ_INCLUDE_PREDICTOR_HH

#include <cstdint>

namespace cusz {

template <typename T, typename E>
class PredictorAbstraction {
   public:
    using Origin  = T;
    using Anchor  = T;
    using ErrCtrl = E;

   private:
    void partition_workspace();

   public:
    // helper functions
    virtual uint32_t get_workspace_nbyte() const = 0;

    virtual uint32_t get_data_len() const    = 0;
    virtual uint32_t get_quant_len() const   = 0;
    virtual uint32_t get_anchor_len() const  = 0;
    virtual uint32_t get_outlier_len() const = 0;

    virtual float get_time_elapsed() const = 0;

    // "real" methods
    virtual ~PredictorAbstraction() = default;

    // methods
    virtual void construct(
        T* const           in_data,
        T* const           out_anchor,
        E* const           out_errctrl,
        double const       eb,
        int const          radius,
        cudaStream_t const stream,
        T* const __restrict__ non_overlap_out_outlier) = 0;

    virtual void reconstruct(
        T* const           in_anchor,
        E* const           in_errctrl,
        T* const           out_xdata,
        double const       eb,
        int const          radius,
        cudaStream_t const stream,
        T* const __restrict__ non_overlap_in_outlier) = 0;
};

}  // namespace cusz

#endif