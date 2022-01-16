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

#include <exception>
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

#define ALLOCMANAGED(VAR, SYM, NBYTE)   \
    cudaMallocManaged(&d_##VAR, NBYTE); \
    cudaMemset(d_##VAR, 0x0, NBYTE);

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
    bool dbg_mode{false};

    unsigned int dimx, dimx_aligned, nblockx, nanchorx;
    unsigned int dimy, dimy_aligned, nblocky, nanchory;
    unsigned int dimz, dimz_aligned, nblockz, nanchorz;
    unsigned int len, len_aligned, len_anchor;
    dim3         size, size_aligned, leap, leap_aligned, anchor_size, anchor_leap;

    bool delay_postquant_dummy;
    bool outlier_overlapped;

    float time_elapsed;

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
     * @deprecated use another construct method instead; will remove when cleaning
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
        throw std::runtime_error("obsolete");
    }

    /**
     * @deprecated use another reconstruct method instread; will remove when cleaning
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
        throw std::runtime_error("obsolete");
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

    Spline3(dim3 _size, bool _dbg_mode = false) : size(_size), dbg_mode(_dbg_mode)
    {
        auto debug = [&]() {
            printf("\ndebug in spline3::constructor\n");
            printf("dim.xyz & len:\t%d, %d, %d, %d\n", dimx, dimy, dimz, len);
            printf("nblock.xyz:\t%d, %d, %d\n", nblockx, nblocky, nblockz);
            printf("aligned.xyz:\t%d, %d, %d\n", dimx_aligned, dimy_aligned, dimz_aligned);
            printf("nanchor.xyz:\t%d, %d, %d\n", nanchorx, nanchory, nanchorz);
            printf("data_len:\t%d\n", get_data_len());
            printf("anchor_len:\t%d\n", get_anchor_len());
            printf("quant_len:\t%d\n", get_quant_len());
            printf("quant_footprint:\t%d\n", get_quant_footprint());
            printf("NBYTE anchor:\t%lu\n", sizeof(T) * len_anchor);
            printf("NBYTE errctrl:\t%lu\n", sizeof(E) * get_quant_footprint());
            cout << '\n';
        };

        // original size
        dimx = size.x, dimy = size.y, dimz = size.z;
        len = dimx * dimy * dimz;

        // partition & aligning
        nblockx      = ConfigHelper::get_npart(dimx, BLOCK * 4);
        nblocky      = ConfigHelper::get_npart(dimy, BLOCK);
        nblockz      = ConfigHelper::get_npart(dimz, BLOCK);
        dimx_aligned = nblockx * 32;  // 235 -> 256
        dimy_aligned = nblocky * 8;   // 449 -> 456
        dimz_aligned = nblockz * 8;   // 449 -> 456
        len_aligned  = dimx_aligned * dimy_aligned * dimz_aligned;

        // multidimensional
        leap         = dim3(1, dimx, dimx * dimy);
        size_aligned = dim3(dimx_aligned, dimy_aligned, dimz_aligned);
        leap_aligned = dim3(1, dimx_aligned, dimx_aligned * dimy_aligned);

        // anchor point
        nanchorx    = int(dimx / BLOCK) + 1;
        nanchory    = int(dimy / BLOCK) + 1;
        nanchorz    = int(dimz / BLOCK) + 1;
        len_anchor  = nanchorx * nanchory * nanchorz;
        anchor_size = dim3(nanchorx, nanchory, nanchorz);
        anchor_leap = dim3(1, nanchorx, nanchorx * nanchory);

        if (dbg_mode) debug();
    }

    /**
     * @brief Allocate workspace according to the input size.
     *
     * @param xyz (host variable) 3D size for input data
     * @param dbg_managed (host variable) use unified memory for debugging
     * @param _delay_postquant_dummy (host variable) (future) control the delay of postquant
     * @param _outlier_overlapped (host variable) (future) control the input-output overlapping
     */
    void allocate_workspace(bool _delay_postquant_dummy = false, bool _outlier_overlapped = true)
    {
        // config
        delay_postquant_dummy = _delay_postquant_dummy;
        outlier_overlapped    = _outlier_overlapped;

        // allocate
        auto nbyte_anchor = sizeof(T) * get_anchor_len();
        printf("nbyte_anchor: %lu\n", nbyte_anchor);
        cudaMalloc(&d_anchor, nbyte_anchor);
        cudaMemset(d_anchor, 0x0, nbyte_anchor);

        auto nbyte_errctrl = sizeof(E) * get_quant_footprint();
        printf("nbyte_errctrl: %lu\n", nbyte_errctrl);
        cudaMalloc(&d_errctrl, nbyte_errctrl);
        cudaMemset(d_errctrl, 0x0, nbyte_errctrl);

        if (not outlier_overlapped) {
            auto nbyte_outlier = sizeof(T) * get_quant_footprint();
            cudaMalloc(&d_outlier, nbyte_outlier);
            cudaMemset(d_outlier, 0x0, nbyte_outlier);
        }
    }

    ~Spline3()
    {
        FREEDEV(anchor);
        FREEDEV(errctrl);
    }

    /**
     * @brief Construct error-control code & outlier; input and outlier overlap each other. Thus, it's destructive.
     *
     * @param in_data (device array) input data and output outlier
     * @param cfg_eb (host variable) error bound; configuration
     * @param cfg_radius (host variable) radius to control the bound; configuration
     * @param ptr_anchor (device array) output anchor point
     * @param ptr_errctrl (device array) output error-control code; if range-limited integer, it is quant-code
     * @param stream CUDA stream
     */
    void construct(
        TITER        in_data,
        double const eb,
        int const    radius,
        TITER&       out_anchor,
        EITER&       out_errctrl,
        cudaStream_t stream = nullptr)
    {
        auto ebx2 = eb * 2;
        auto eb_r = 1 / eb;

        out_anchor  = d_anchor;
        out_errctrl = d_errctrl;

        if (dbg_mode) {
            printf("\nSpline3::construct dbg:\n");
            printf("ebx2: %lf\n", ebx2);
            printf("eb_r: %lf\n", eb_r);
        }

        cuda_timer_t timer;
        timer.timer_start();

        cusz::c_spline3d_infprecis_32x8x8data<TITER, EITER, float, 256, false>
            <<<dim3(nblockx, nblocky, nblockz), dim3(256, 1, 1), 0, stream>>>  //
            (in_data, size, leap,                                              //
             d_errctrl, size_aligned, leap_aligned,                            //
             d_anchor, anchor_leap,                                            //
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
        TITER        in_outlier__out_xdata,
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
             eb_r, ebx2, cfg_radius);

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

#undef FREEMANAGED
#undef ALLOCMANAGED

#undef DEFINE_ARRAY

#endif
