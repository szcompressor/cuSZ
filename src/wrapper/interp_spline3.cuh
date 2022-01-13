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

#include <limits>
#include <numeric>

#include "../../include/predictor.hh"
#include "../common.hh"
#include "../kernel/spline3.cuh"
#include "../utils.hh"

#define DEFINE_ARRAY(VAR, TYPE) TYPE* d_##VAR{nullptr};

#define ALLOCDEV(VAR, SYM, NBYTE) \
    cudaMalloc(&d_##VAR, NBYTE);  \
    cudaMemset(d_##VAR, 0x0, NBYTE);

#define FREEDEV(VAR)       \
    if (d_##VAR) {         \
        cudaFree(d_##VAR); \
        d_##VAR = nullptr; \
    }

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

    /**
     * @brief Constructor
     * @deprecated use default constructor instead
     *
     * @param xyz
     * @param _eb
     * @param _radius
     * @param _delay_postquant_dummy
     */
    Spline3(dim3 xyz, double _eb, int _radius = 0, bool _delay_postquant_dummy = true);

    /**
     * @brief
     * @deprecated use another construct method instead; will remove when cleaning
     *
     * @param in_data
     * @param out_anchor
     * @param out_errctrl
     * @param eb
     * @param radius
     * @param non_overlap_out_outlier
     */
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

    /**
     * @brief
     * @deprecated use another reconstruct method instread; will remove when cleaning
     *
     * @param in_anchor
     * @param in_errctrl
     * @param out_data
     * @param eb
     * @param radius
     * @param non_overlap_in_outlier
     */
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
    DEFINE_ARRAY(anchor, T);
    DEFINE_ARRAY(errctrl, E);
    DEFINE_ARRAY(outlier, T);

   public:
    E* expose_quant() const { return d_errctrl; }
    E* expose_errctrl() const { return d_errctrl; }
    T* expose_anchor() const { return d_anchor; }

   public:
    Spline3() = default;

    /**
     * @brief Allocate workspace according to the input size.
     *
     * @param xyz (host variable) 3D size for input data
     * @param _delay_postquant_dummy (host variable) (future) control the delay of postquant
     * @param _outlier_overlapped (host variable) (future) control the input-output overlapping
     */
    void allocate_workspace(dim3 xyz, bool _delay_postquant_dummy = false, bool _outlier_overlapped = true)
    {
        auto debug = [&]() {
            cout << '\n';
            cout << "debug in spline3::allocate_workspace\n";
            cout << "get_data_len: " << get_data_len() << '\n';
            cout << "get_anchor_len " << get_anchor_len() << '\n';
            cout << "get_quant_len " << get_quant_len() << '\n';
            cout << "get_quant_footprint: " << get_quant_footprint() << '\n';
            cout << '\n';
        };

        size = xyz;
        dimx = xyz.x, dimy = xyz.y, dimz = xyz.z;
        len = dimx * dimy * dimz;

        delay_postquant_dummy = _delay_postquant_dummy;
        outlier_overlapped    = _outlier_overlapped;

        // data size
        nblockx = ConfigHelper::get_npart(dimx, BLOCK * 4);
        nblocky = ConfigHelper::get_npart(dimy, BLOCK);
        nblockz = ConfigHelper::get_npart(dimz, BLOCK);
        // (235, 449, 449) -> (256, 456, 456)
        dimx_aligned = nblockx * 32, dimy_aligned = nblocky * 8, dimz_aligned = nblockz * 8;
        len_aligned = dimx_aligned * dimy_aligned * dimz_aligned;

        leap         = dim3(1, dimx, dimx * dimy);
        size_aligned = dim3(dimx_aligned, dimy_aligned, dimz_aligned);
        leap_aligned = dim3(1, dimx_aligned, dimx_aligned * dimy_aligned);

        // anchor point
        nanchorx   = int(dimx / BLOCK) + 1;
        nanchory   = int(dimy / BLOCK) + 1;
        nanchorz   = int(dimz / BLOCK) + 1;
        len_anchor = nanchorx * nanchory * nanchorz;

        // 1D
        anchor_size = dim3(nanchorx, nanchory, nanchorz);
        anchor_leap = dim3(1, nanchorx, nanchorx * nanchory);

        // allocate
        {
            auto nbyte = sizeof(T) * len_anchor;
            ALLOCDEV(anchor, T, nbyte);  // for lorenzo, anchor can be 0
        }
        {
            auto nbyte = sizeof(E) * get_quant_footprint();
            ALLOCDEV(errctrl, E, nbyte);
        }
        if (not outlier_overlapped) {
            auto nbyte = sizeof(E) * get_quant_footprint();
            ALLOCDEV(outlier, T, nbyte);
        }

        debug();
    }

    ~Spline3()
    {
        FREEDEV(anchor);
        FREEDEV(errctrl);
    }

    /**
     * @brief Construct error-control code & outlier; input and outlier overlap each other. Thus, it's destructive.
     *
     * @param in_data__out_outlier (device array) input data and output outlier
     * @param cfg_eb (host variable) error bound; configuration
     * @param cfg_radius (host variable) radius to control the bound; configuration
     * @param out_anchor (device array) output anchor point
     * @param out_errctrl (device array) output error-control code; if range-limited integer, it is quant-code
     * @param stream CUDA stream
     */
    void construct(
        TITER        in_data__out_outlier,
        double const cfg_eb,
        int const    cfg_radius,
        TITER&       out_anchor,
        EITER&       out_errctrl,
        cudaStream_t stream = nullptr)
    {
        auto ebx2 = eb * 2;
        auto eb_r = 1 / eb;

        out_anchor  = d_anchor;
        out_errctrl = d_errctrl;

        cuda_timer_t timer;
        timer.timer_start();

        cusz::c_spline3d_infprecis_32x8x8data<TITER, EITER, float, 256, false>
            <<<dim3(nblockx, nblocky, nblockz), dim3(256, 1, 1), 0, stream>>>  //
            (in_data__out_outlier, size, leap,                                 //
             out_errctrl, size_aligned, leap_aligned,                          //
             out_anchor, anchor_leap,                                          //
             eb_r, ebx2, radius);

        timer.timer_end();

        if (stream)
            CHECK_CUDA(cudaStreamSynchronize(stream));
        else
            CHECK_CUDA(cudaDeviceSynchronize());

        time_elapsed = timer.get_time_elapsed();
    }

    /**
     * @brief Reconstruct data from error-control code & outlier; outlier and output overlap each other; destructive for
     * outlier.
     *
     * @param in_anchor (device array) input anchor
     * @param in_errctrl (device array) input error-control code
     * @param cfg_eb (host variable) error bound; configuration
     * @param cfg_radius (host variable) radius to control the bound; configuration
     * @param in_outlier__out_xdata (device array) output reconstructed data, overlapped with input outlier
     * @param stream CUDA stream
     */
    void reconstruct(
        TITER        in_anchor,
        EITER        in_errctrl,
        double const cfg_eb,
        int const    cfg_radius,
        TITER&       in_outlier__out_xdata,
        cudaStream_t stream = nullptr)
    {
        auto ebx2 = cfg_eb * 2;
        auto eb_r = 1 / cfg_eb;

        cuda_timer_t timer;
        timer.timer_start();

        cusz::x_spline3d_infprecis_32x8x8data<EITER, TITER, float, 256>
            <<<dim3(nblockx, nblocky, nblockz), dim3(256, 1, 1), 0, stream>>>  //
            (in_errctrl, size_aligned, leap_aligned,                           //
             in_anchor, anchor_size, anchor_leap,                              //
             in_outlier__out_xdata, size, leap,                                //
             eb_r, ebx2, radius);

        timer.timer_end();

        if (stream)
            CHECK_CUDA(cudaStreamSynchronize(stream));
        else
            CHECK_CUDA(cudaDeviceSynchronize());

        time_elapsed = timer.get_time_elapsed();
    }

    // end of class definition
};

}  // namespace cusz

#undef FREEDEV
#undef ALLOCDEV
#undef DEFINE_ARRAY

#endif
