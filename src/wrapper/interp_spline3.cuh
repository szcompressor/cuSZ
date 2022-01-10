/**
 * @file interp_spline3.cuh
 * @author Jiannan Tian
 * @brief (header) A high-level Spline3D wrapper. Allocations are explicitly out of called functions.
 * @version 0.3
 * @date 2021-06-15
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_WRAPPER_INTERP_SPLINE_CUH
#define CUSZ_WRAPPER_INTERP_SPLINE_CUH

#include "../../include/predictor.hh"
#include "../common.hh"

namespace cusz {

template <typename T, typename E, typename FP>
class Spline3 : public PredictorAbstraction<T, E> {
   public:
    using Precision = FP;

   private:
    static const auto BLOCK = 8;

    using TITER = T*;
    using EITER = E*;

   private:
    unsigned int dimx, dimx_aligned, nblockx, nanchorx;
    unsigned int dimy, dimy_aligned, nblocky, nanchory;
    unsigned int dimz, dimz_aligned, nblockz, nanchorz;
    unsigned int len, len_aligned, len_anchor;
    dim3         size, size_aligned, leap, leap_aligned, anchor_size, anchor_leap;

    double eb;
    FP     eb_r, ebx2, ebx2_r;

    bool delay_postquant_dummy;
    bool outlier_overlapped;

    float time_elapsed;

    int radius{0};

   public:
    unsigned int get_data_len() const { return len; }
    unsigned int get_anchor_len() const { return len_anchor; }
    unsigned int get_quant_len() const
    {
        auto m = Reinterpret1DTo2D::get_square_size(len_aligned);
        return m * m;
    }

    // TODO this is just a placehodler
    unsigned int get_outlier_len() const
    {
        throw std::runtime_error("spline3::get_outlier_len() not implemented");
        return 0;
    }

    unsigned int get_quant_footprint() const
    {
        auto m = Reinterpret1DTo2D::get_square_size(len_aligned);
        return m * m;
    }
    float    get_time_elapsed() const { return time_elapsed; }
    uint32_t get_workspace_nbyte() const { return 0; };

    Spline3(dim3 xyz, double _eb, int _radius = 0, bool _delay_postquant_dummy = true);

    void construct(
        TITER        in_data,
        TITER        out_anchor,
        EITER        out_errctrl,
        double const eb,
        int const    radius,
        cudaStream_t                                  = nullptr,
        T* const __restrict__ non_overlap_out_outlier = nullptr)
    {
        ////////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////  obsolete  //////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////
    }

    void reconstruct(
        TITER        in_anchor,
        EITER        in_errctrl,
        TITER        out_data,
        double const eb,
        int const    radius,
        cudaStream_t                                 = nullptr,
        T* const __restrict__ non_overlap_in_outlier = nullptr)
    {
        ////////////////////////////////////////////////////////////////////////////////
        //////////////////////////////////  obsolete  //////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////
    }

    // new ------------------------------------------------------------
   private:
#define DEFINE_ARRAY(VAR, TYPE) TYPE* d_##VAR{nullptr};
    DEFINE_ARRAY(anchor, T);
    DEFINE_ARRAY(errctrl, E);
    DEFINE_ARRAY(outlier, T);
#undef DEFINE_ARRAY

   public:
    E* expose_quant() const { return d_errctrl; }
    E* expose_errctrl() const { return d_errctrl; }
    T* expose_anchor() const { return d_anchor; }

   public:
    Spline3() = default;

    void allocate_workspace(dim3 _size3, bool _delay_postquant = false, bool _outlier_overlapped = true);

    ~Spline3()
    {
#define SPLINE3_FREEDEV(VAR) \
    if (d_##VAR) {           \
        cudaFree(d_##VAR);   \
        d_##VAR = nullptr;   \
    }
        SPLINE3_FREEDEV(anchor);
        SPLINE3_FREEDEV(errctrl);

#undef SPLINE3_FREEDEV
    }

    void construct(
        TITER        in_data,
        double const cfg_eb,
        int const    cfg_radius,
        TITER&       out_anchor,
        EITER&       out_errctrl,
        cudaStream_t stream                           = nullptr,
        T* const __restrict__ non_overlap_out_outlier = nullptr);

    void reconstruct(
        TITER        in_anchor,
        EITER        in_errctrl,
        double const cfg_eb,
        int const    cfg_radius,
        TITER&       out_data,
        cudaStream_t stream                          = nullptr,
        T* const __restrict__ non_overlap_in_outlier = nullptr);
};

}  // namespace cusz

#endif
