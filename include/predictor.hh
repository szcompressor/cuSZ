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
    virtual size_t get_workspace_nbyte() const = 0;

    virtual size_t get_len_data() const    = 0;
    virtual size_t get_len_quant() const   = 0;
    virtual size_t get_len_anchor() const  = 0;
    virtual size_t get_len_outlier() const = 0;

    virtual float get_time_elapsed() const = 0;

    // "real" methods
    virtual ~PredictorAbstraction() = default;

    /*
    virtual void construct(
        T* const           in_data,
        T* const           out_anchor,
        E* const           out_errctrl,
        dim3               base_len3,
        double const       eb,
        int const          radius,
        cudaStream_t const stream) = 0;

    virtual void reconstruct(
        T* const           in_anchor,
        E* const           in_errctrl,
        T* const           out_xdata,
        dim3               base_len3,
        double const       eb,
        int const          radius,
        cudaStream_t const stream) = 0;
    */
};

}  // namespace cusz

#endif
