/**
 * @file extrap_lorenzo.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-06-16
 * (rev.1) 2021-09-18 (rev.2) 2022-01-10
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_WRAPPER_EXTRAP_LORENZO_CUH
#define CUSZ_WRAPPER_EXTRAP_LORENZO_CUH

#include <clocale>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "../../include/predictor.hh"

#include "../common.hh"
#include "../utils.hh"

#ifdef DPCPP_SHOWCASE
#include "../kernel/lorenzo_prototype.cuh"

using cusz::prototype::c_lorenzo_1d1l;
using cusz::prototype::c_lorenzo_2d1l;
using cusz::prototype::c_lorenzo_3d1l;
using cusz::prototype::x_lorenzo_1d1l;
using cusz::prototype::x_lorenzo_2d1l;
using cusz::prototype::x_lorenzo_3d1l;

#else
#include "../kernel/lorenzo.cuh"
#endif

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

#define ALLOCDEV(VAR, SYM, NBYTE)                    \
    if (NBYTE != 0) {                                \
        CHECK_CUDA(cudaMalloc(&d_##VAR, NBYTE));     \
        CHECK_CUDA(cudaMemset(d_##VAR, 0x0, NBYTE)); \
    }

#define FREE_DEV_ARRAY(VAR)            \
    if (d_##VAR) {                     \
        CHECK_CUDA(cudaFree(d_##VAR)); \
        d_##VAR = nullptr;             \
    }

#define DEFINE_ARRAY(VAR, TYPE) TYPE* d_##VAR{nullptr};

namespace cusz {

template <typename T, typename E, typename FP>
class PredictorLorenzo : public PredictorAbstraction<T, E> {
   public:
    using Precision = FP;

   private:
    dim3 size;  // size.x, size.y, size.z
    dim3 leap;  // 1, leap.y, leap.z
    int  ndim;
    // may differ from `len_data`
    uint32_t len_data, len_outlier, len_quant;
    bool     delay_postquant;
    bool     outlier_overlapped;

    float time_elapsed;

    // future use
    // struct {
    //     bool count_nnz;
    //     bool blockwide_gather;
    // } on_off;

    /**
     * @brief Construction dispatcher; handling 1,2,3-D;
     *
     * @tparam DELAY_POSTQUANT (compile time) control if delaying postquant
     * @param in_data (device array) input data; possibly overlapping with output outlier
     * @param out_anchor (device array) output anchor
     * @param out_errctrl (device array) output error-control code; if range-limited integer, it is quant-code
     * @param __out_outlier (device array) non-overlapping (with input data) outlier
     * @param eb (host variable) error bound; configuration
     * @param radius (host variable) radius to control the bound; configuration
     * @param stream CUDA stream
     */
    template <bool DELAY_POSTQUANT>
    void construct_proxy(
        T* const in_data,
        T* const out_anchor,
        E* const out_errctrl,
        T* const __restrict__ __out_outlier,
        double const       eb,
        int const          radius,
        cudaStream_t const stream = nullptr)
    {
        // error bound
        auto ebx2   = eb * 2;
        auto ebx2_r = 1 / ebx2;

        // decide if destructive for the input (data)
        auto out_outlier = __out_outlier == nullptr ? in_data : __out_outlier;

        // TODO put into conditional compile
        cuda_timer_t timer;
        timer.timer_start(stream);

        if (ndim == 1) {
            constexpr auto SEQ          = 4;
            constexpr auto DATA_SUBSIZE = 256;
            auto           dim_block    = DATA_SUBSIZE / SEQ;
            auto           dim_grid     = ConfigHelper::get_npart(size.x, DATA_SUBSIZE);
            ::cusz::c_lorenzo_1d1l<T, E, FP, DATA_SUBSIZE, SEQ, DELAY_POSTQUANT>  //
                <<<dim_grid, dim_block, 0, stream>>>                              //
                (in_data, out_errctrl, out_outlier, size.x, radius, ebx2_r);
        }
        else if (ndim == 2) {  // y-sequentiality == 8
            auto dim_block = dim3(16, 2);
            auto dim_grid  = dim3(ConfigHelper::get_npart(size.x, 16), ConfigHelper::get_npart(size.y, 16));
            ::cusz::c_lorenzo_2d1l_16x16data_mapto16x2<T, E, FP>  //
                <<<dim_grid, dim_block, 0, stream>>>              //
                (in_data, out_errctrl, out_outlier, size.x, size.y, leap.y, radius, ebx2_r);
        }
        else if (ndim == 3) {  // y-sequentiality == 8
            auto dim_block = dim3(32, 1, 8);
            auto dim_grid  = dim3(
                 ConfigHelper::get_npart(size.x, 32), ConfigHelper::get_npart(size.y, 8),
                 ConfigHelper::get_npart(size.z, 8));
            ::cusz::c_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, E, FP>  //
                <<<dim_grid, dim_block, 0, stream>>>                 //
                (in_data, out_errctrl, out_outlier, size.x, size.y, size.z, leap.y, leap.z, radius, ebx2_r);
        }
        else {
            throw std::runtime_error("Lorenzo only works for 123-D.");
        }

        timer.timer_end(stream);
        if (stream)
            CHECK_CUDA(cudaStreamSynchronize(stream));
        else
            CHECK_CUDA(cudaDeviceSynchronize());

        time_elapsed = timer.get_time_elapsed();
    }

    /**
     * @brief Reconstruction dispatcher; handling 1,2,3-D;
     *
     * @tparam DELAY_POSTQUANT (compile time) control if delaying postquant
     * @param __in_outlier (device array) non-overlapping (with output xdata) outlier
     * @param in_anchor (device array) input anchor
     * @param in_errctrl (device array) input error-control code; if range-limited integer, it is quant-code
     * @param out_xdata (device array) output reconstructed data; possibly overlapping with input outlier
     * @param eb (host variable) error bound; configuration
     * @param radius (host variable) radius to control the bound; configuration
     * @param stream CUDA stream
     */
    template <bool DELAY_POSTQUANT>
    void reconstruct_proxy(
        T* const __restrict__ __in_outlier,
        T* const           in_anchor,
        E* const           in_errctrl,
        T* const           out_xdata,
        double const       eb,
        int const          radius,
        cudaStream_t const stream = nullptr)
    {
        // error bound
        auto ebx2   = eb * 2;
        auto ebx2_r = 1 / ebx2;

        // decide if destructive for the input (outlier)
        auto in_outlier = __in_outlier == nullptr ? out_xdata : __in_outlier;

        cuda_timer_t timer;
        timer.timer_start(stream);

        if (ndim == 1) {  // y-sequentiality == 8
            constexpr auto SEQ          = 8;
            constexpr auto DATA_SUBSIZE = 256;
            auto           dim_block    = DATA_SUBSIZE / SEQ;
            auto           dim_grid     = ConfigHelper::get_npart(size.x, DATA_SUBSIZE);
            ::cusz::x_lorenzo_1d1l<T, E, FP, DATA_SUBSIZE, SEQ, DELAY_POSTQUANT>  //
                <<<dim_grid, dim_block, 0, stream>>>                              //
                (in_outlier, in_errctrl, out_xdata, size.x, radius, ebx2);
        }
        else if (ndim == 2) {  // y-sequentiality == 8
            auto dim_block = dim3(16, 2);
            auto dim_grid  = dim3(ConfigHelper::get_npart(size.x, 16), ConfigHelper::get_npart(size.y, 16));

            ::cusz::x_lorenzo_2d1l_16x16data_mapto16x2<T, E, FP, DELAY_POSTQUANT>  //
                <<<dim_grid, dim_block, 0, stream>>>                               //
                (in_outlier, in_errctrl, out_xdata, size.x, size.y, leap.y, radius, ebx2);
        }
        else if (ndim == 3) {  // y-sequentiality == 8
            auto dim_block = dim3(32, 1, 8);
            auto dim_grid  = dim3(
                 ConfigHelper::get_npart(size.x, 32), ConfigHelper::get_npart(size.y, 8),
                 ConfigHelper::get_npart(size.z, 8));

            ::cusz::x_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, E, FP, DELAY_POSTQUANT>  //
                <<<dim_grid, dim_block, 0, stream>>>                                  //
                (in_outlier, in_errctrl, out_xdata, size.x, size.y, size.z, leap.y, leap.z, radius, ebx2);
        }

        timer.timer_end(stream);
        if (stream)
            CHECK_CUDA(cudaStreamSynchronize(stream));
        else
            CHECK_CUDA(cudaDeviceSynchronize());

        time_elapsed = timer.get_time_elapsed();
    }

   public:
    // constructor
    PredictorLorenzo() = default;

    /**
     * @brief Construct a new Predictor Lorenzo object
     * @deprecated use the default constructor and `allocate_workspace` instead
     *
     * @param _size
     * @param delay_postquant
     */
    PredictorLorenzo(dim3 _size, bool _delay_postquant = false, bool _outlier_overlapped = true) : size(_size)
    {
        // size
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
    }

    ~PredictorLorenzo()
    {
        FREE_DEV_ARRAY(anchor);
        FREE_DEV_ARRAY(errctrl);
    };

    // helper
    uint32_t get_data_len() const { return len_data; }
    uint32_t get_anchor_len() const { return 0; }
    uint32_t get_quant_len() const { return len_quant; }
    uint32_t get_outlier_len() const { return len_outlier; }
    uint32_t get_workspace_nbyte() const { return 0; };

    float get_time_elapsed() const { return time_elapsed; }

    /**
     * @brief
     * @deprecated use `construct(in..., cfg..., out..., opt_args...)` instead
     *
     * @param in_data
     * @param out_anchor
     * @param out_errctrl
     * @param eb
     * @param radius
     * @param stream
     * @param non_overlap_out_outlier
     */
    void construct(
        T* const           in_data,
        T* const           out_anchor,
        E* const           out_errctrl,
        double const       eb,
        int const          radius,
        cudaStream_t const stream               = nullptr,
        T* __restrict__ non_overlap_out_outlier = nullptr)
    {
        if (not delay_postquant)
            construct_proxy<false>(in_data, out_anchor, out_errctrl, non_overlap_out_outlier, eb, radius, stream);
        else
            throw std::runtime_error("construct_proxy<delay_postquant==true> not implemented.");
    }

    /**
     * @brief
     * @deprecated use `reconstruct(in..., cfg..., out..., opt_args...)` instead
     *
     * @param in_anchor
     * @param in_errctrl
     * @param out_xdata
     * @param eb
     * @param radius
     * @param stream
     * @param non_overlap_in_outlier
     */
    void reconstruct(
        T* const           in_anchor,
        E* const           in_errctrl,
        T* const           out_xdata,
        double const       eb,
        int const          radius,
        cudaStream_t const stream              = nullptr,
        T* __restrict__ non_overlap_in_outlier = nullptr)
    {
        if (not delay_postquant)
            reconstruct_proxy<false>(non_overlap_in_outlier, in_anchor, in_errctrl, out_xdata, eb, radius, stream);
        else
            throw std::runtime_error("construct_proxy<delay_postquant==true> not implemented.");
    }

    // refactor below
   private:
    DEFINE_ARRAY(anchor, T);
    DEFINE_ARRAY(errctrl, E);
    DEFINE_ARRAY(outlier, T);
#undef DEFINE_ARRAY

   public:
    /**
     * @brief Allocate workspace according to the input size.
     *
     * @param _size3 (host variable) 3D size for input data
     * @param _delay_postquant (host variable) (future) control the delay of postquant
     * @param _outlier_overlapped (host variable) (future) control the input-output overlapping
     */
    void allocate_workspace(bool dbg_print = false)
    {
        auto debug = [&]() {
            setlocale(LC_NUMERIC, "");

            printf("\nPredictorLorenzo::allocate_workspace() debugging:\n");
            printf("%-*s:  (%u, %u, %u)\n", 16, "size.xyz", size.x, size.y, size.z);
            printf("%-*s:  (%u, %u, %u)\n", 16, "leap.xyz", leap.x, leap.y, leap.z);
            printf("%-*s:  (%u, %u, %u)\n", 16, "sizeof.{T,E,FP}", (int)sizeof(T), (int)sizeof(E), (int)sizeof(FP));
            printf("%-*s:  %'u\n", 16, "len.data", len_data);
            printf("%-*s:  %'u\n", 16, "len.quant", len_quant);
            printf("%-*s:  %'u\n", 16, "len.outlier", len_outlier);
        };

        // allocate
        ALLOCDEV(anchor, T, 0);  // for lorenzo, anchor can be 0
        ALLOCDEV(errctrl, E, sizeof(E) * len_quant);
        if (not outlier_overlapped) ALLOCDEV(outlier, T, sizeof(T) * len_data);

        if (dbg_print) debug();
    }

   public:
    E* expose_quant() const { return d_errctrl; }
    E* expose_errctrl() const { return d_errctrl; }
    T* expose_anchor() const { return d_anchor; }
    T* expose_outlier() const { return d_outlier; }

    /**
     * @brief Construct error-control code & outlier; input and outlier do NOT overlap each other.
     *
     * @param in_data (device array) input data
     * @param eb (host variable) error bound; configuration
     * @param radius (host variable) radius to control the bound; configuration
     * @param out_anchor (device array) output anchor point
     * @param out_errctrl (device array) output error-control code; if range-limited integer, it is quant-code
     * @param out_outlier (device array) non-overlapping (with `in_data`) output outlier
     * @param stream CUDA stream
     */
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

    /**
     * @brief Construct error-control code & outlier; input and outlier overlap each other. Thus, it's destructive.
     *
     * @param in_data__out_outlier (device array) input data and output outlier
     * @param eb (host variable) error bound; configuration
     * @param radius (host variable) radius to control the bound; configuration
     * @param out_anchor (device array) output anchor point
     * @param out_errctrl (device array) output error-control code; if range-limited integer, it is quant-code
     * @param stream CUDA stream
     */
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

    /**
     * @brief Reconstruct data from error-control code & outlier; outlier and output do NOT overlap each other.
     *
     * @param in_outlier (device array) input outlier
     * @param in_anchor (device array) input anchor
     * @param in_errctrl (device array) input error-control code
     * @param eb (host variable) error bound; configuration
     * @param radius (host variable) radius to control the bound; configuration
     * @param out_xdata (device array) reconstructed data; output
     * @param stream CUDA stream
     */
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

    /**
     * @brief Reconstruct data from error-control code & outlier; outlier and output overlap each other; destructive for
     * outlier.
     *
     * @param in_anchor (device array) input anchor
     * @param in_errctrl (device array) input error-control code
     * @param eb (host variable) error bound; configuration
     * @param radius (host variable) radius to control the bound; configuration
     * @param in_outlier__out_xdata (device array) output reconstructed data, overlapped with input outlier
     * @param stream CUDA stream
     */
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

    // end of class
};

}  // namespace cusz

#undef ALLOCDEV
#undef FREE_DEV_ARRAY
#undef DEFINE_ARRAY

#endif