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

#define DEFINE_ARRAY(VAR, TYPE) TYPE* d_##VAR{nullptr};

namespace cusz {

template <typename T, typename E, typename FP>
class PredictorLorenzo : public PredictorAbstraction<T, E> {
   public:
    using Precision = FP;

   private:
    bool delay_postquant{false};
    bool outlier_overlapped{true};

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

        if (rt_len.ndim == 1) {
            constexpr auto SEQ         = 4;
            constexpr auto DATA_SUBLEN = 256;
            auto           dim_block   = DATA_SUBLEN / SEQ;
            auto           dim_grid    = ConfigHelper::get_npart(rt_len.len3.x, DATA_SUBLEN);
            ::cusz::c_lorenzo_1d1l<T, E, FP, DATA_SUBLEN, SEQ, DELAY_POSTQUANT>  //
                <<<dim_grid, dim_block, 0, stream>>>                             //
                (in_data, out_errctrl, out_outlier, rt_len.len3.x, radius, ebx2_r);
        }
        else if (rt_len.ndim == 2) {  // y-sequentiality == 8
            auto dim_block = dim3(16, 2);
            auto dim_grid =
                dim3(ConfigHelper::get_npart(rt_len.len3.x, 16), ConfigHelper::get_npart(rt_len.len3.y, 16));
            ::cusz::c_lorenzo_2d1l_16x16data_mapto16x2<T, E, FP>  //
                <<<dim_grid, dim_block, 0, stream>>>              //
                (in_data, out_errctrl, out_outlier, rt_len.len3.x, rt_len.len3.y, rt_len.leap.y, radius, ebx2_r);
        }
        else if (rt_len.ndim == 3) {  // y-sequentiality == 8
            auto dim_block = dim3(32, 1, 8);
            auto dim_grid  = dim3(
                 ConfigHelper::get_npart(rt_len.len3.x, 32), ConfigHelper::get_npart(rt_len.len3.y, 8),
                 ConfigHelper::get_npart(rt_len.len3.z, 8));
            ::cusz::c_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, E, FP>  //
                <<<dim_grid, dim_block, 0, stream>>>                 //
                (in_data, out_errctrl, out_outlier, rt_len.len3.x, rt_len.len3.y, rt_len.len3.z, rt_len.leap.y,
                 rt_len.leap.z, radius, ebx2_r);
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

        if (rt_len.ndim == 1) {  // y-sequentiality == 8
            constexpr auto SEQ         = 8;
            constexpr auto DATA_SUBLEN = 256;
            auto           dim_block   = DATA_SUBLEN / SEQ;
            auto           dim_grid    = ConfigHelper::get_npart(rt_len.len3.x, DATA_SUBLEN);
            ::cusz::x_lorenzo_1d1l<T, E, FP, DATA_SUBLEN, SEQ, DELAY_POSTQUANT>  //
                <<<dim_grid, dim_block, 0, stream>>>                             //
                (in_outlier, in_errctrl, out_xdata, rt_len.len3.x, radius, ebx2);
        }
        else if (rt_len.ndim == 2) {  // y-sequentiality == 8
            auto dim_block = dim3(16, 2);
            auto dim_grid =
                dim3(ConfigHelper::get_npart(rt_len.len3.x, 16), ConfigHelper::get_npart(rt_len.len3.y, 16));

            ::cusz::x_lorenzo_2d1l_16x16data_mapto16x2<T, E, FP, DELAY_POSTQUANT>  //
                <<<dim_grid, dim_block, 0, stream>>>                               //
                (in_outlier, in_errctrl, out_xdata, rt_len.len3.x, rt_len.len3.y, rt_len.leap.y, radius, ebx2);
        }
        else if (rt_len.ndim == 3) {  // y-sequentiality == 8
            auto dim_block = dim3(32, 1, 8);
            auto dim_grid  = dim3(
                 ConfigHelper::get_npart(rt_len.len3.x, 32), ConfigHelper::get_npart(rt_len.len3.y, 8),
                 ConfigHelper::get_npart(rt_len.len3.z, 8));

            ::cusz::x_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, E, FP, DELAY_POSTQUANT>  //
                <<<dim_grid, dim_block, 0, stream>>>                                  //
                (in_outlier, in_errctrl, out_xdata, rt_len.len3.x, rt_len.len3.y, rt_len.len3.z, rt_len.leap.y,
                 rt_len.leap.z, radius, ebx2);
        }

        timer.timer_end(stream);
        if (stream)
            CHECK_CUDA(cudaStreamSynchronize(stream));
        else
            CHECK_CUDA(cudaDeviceSynchronize());

        time_elapsed = timer.get_time_elapsed();
    }

   private:
    void derive_and_set_alloc_len(dim3 len3)
    {
        alloc_len.data    = len3.x * len3.y * len3.z;
        alloc_len.quant   = alloc_len.data;
        alloc_len.outlier = alloc_len.data;
        alloc_len.anchor  = 0;  // to update
    }

    void derive_and_set_rt_len(dim3 len3)
    {
        rt_len.len3    = len3;
        rt_len.leap    = dim3(1, len3.x, len3.x * len3.y);
        rt_len.data    = len3.x * len3.y * len3.z;
        rt_len.quant   = rt_len.data;
        rt_len.outlier = rt_len.data;
        rt_len.anchor  = 0;

        rt_len.ndim = 3;
        if (len3.z == 1) rt_len.ndim = 2;
        if (len3.z == 1 and len3.y == 1) rt_len.ndim = 1;
    }

   public:
    // constructor
    PredictorLorenzo() = default;

    ~PredictorLorenzo()
    {
        FREE_DEV_ARRAY(anchor);
        FREE_DEV_ARRAY(errctrl);
    };

    // helper
    size_t get_alloclen_data() const { return alloc_len.data; }
    size_t get_alloclen_anchor() const { return alloc_len.anchor; }
    size_t get_alloclen_quant() const { return alloc_len.quant; }
    size_t get_alloclen_outlier() const { return alloc_len.outlier; }

    size_t get_len_data() const { return rt_len.data; }
    size_t get_len_anchor() const { return rt_len.anchor; }
    size_t get_len_quant() const { return rt_len.quant; }
    size_t get_len_outlier() const { return rt_len.outlier; }

    size_t get_workspace_nbyte() const { return 0; };

    float get_time_elapsed() const { return time_elapsed; }

    // refactor below
   private:
    DEFINE_ARRAY(anchor, T);
    DEFINE_ARRAY(errctrl, E);
    DEFINE_ARRAY(outlier, T);
#undef DEFINE_ARRAY

   private:
    struct {
        dim3   len3;
        size_t data, quant, outlier, anchor;
    } alloc_len;

    struct {
        dim3   len3, leap;
        int    ndim;
        size_t data, quant, outlier, anchor;
    } rt_len;

    void debug_list_alloc_len()
    {
        printf("\nPredictorLorenzo::debug_list_alloc_len() debugging:\n");
        printf("%-*s:  (%u, %u, %u)\n", 16, "sizeof.{T,E,FP}", (int)sizeof(T), (int)sizeof(E), (int)sizeof(FP));
        printf("%-*s:  %'u\n", 16, "len.data", alloc_len.data);
        printf("%-*s:  %'u\n", 16, "len.quant", alloc_len.quant);
        printf("%-*s:  %'u\n", 16, "len.outlier", alloc_len.outlier);
    }

    void debug_list_rt_len()
    {
        printf("\nPredictorLorenzo::debug_list_rt_len() debugging:\n");
        printf("%-*s:  (%u, %u, %u)\n", 16, "len3.xyz", rt_len.len3.x, rt_len.len3.y, rt_len.len3.z);
        printf("%-*s:  (%u, %u, %u)\n", 16, "leap.xyz", rt_len.leap.x, rt_len.leap.y, rt_len.leap.z);
        printf("%-*s:  (%u, %u, %u)\n", 16, "sizeof.{T,E,FP}", (int)sizeof(T), (int)sizeof(E), (int)sizeof(FP));
        printf("%-*s:  %'u\n", 16, "len.data", rt_len.data);
        printf("%-*s:  %'u\n", 16, "len.quant", rt_len.quant);
        printf("%-*s:  %'u\n", 16, "len.outlier", rt_len.outlier);
    }

   public:
    /**
     * @brief clear GPU buffer, may affect performance; essentially for debugging
     *
     */
    void clear_buffer() { cudaMemset(d_errctrl, 0x0, sizeof(E) * rt_len.quant); }

    /**
     * @brief Allocate workspace according to the input size.
     *
     * @param x intepreted dimension x
     * @param y intepreted dimension y
     * @param z intepreted dimension z
     * @param dbg_print if enabling debugging print
     */
    void init(size_t x, size_t y, size_t z, bool dbg_print = false)
    {
        auto len3 = dim3(x, y, z);
        init(len3, dbg_print);
    }

    /**
     * @brief Allocate workspace according to the input size.
     *
     * @param xyz interpreted dimensions x, y, z
     * @param dbg_print if enabling debugging print
     */
    void init(dim3 xyz, bool dbg_print = false)
    {
        derive_and_set_alloc_len(xyz);

        // allocate
        ALLOCDEV2(anchor, T, alloc_len.anchor);
        ALLOCDEV2(errctrl, E, alloc_len.quant);
        if (not outlier_overlapped) ALLOCDEV(outlier, T, alloc_len.outlier);

        if (dbg_print) debug_list_alloc_len();
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
        dim3 const   interpreted_len3,
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

        derive_and_set_rt_len(interpreted_len3);

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
        dim3 const         interpreted_len3,
        double const       eb,
        int const          radius,
        T*&                out_anchor,
        E*&                out_errctrl,
        cudaStream_t const stream = nullptr)
    {
        out_anchor  = d_anchor;
        out_errctrl = d_errctrl;

        derive_and_set_rt_len(interpreted_len3);

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
        dim3 interpreted_len3,
        T* __restrict__ in_outlier,
        T*           in_anchor,
        E*           in_errctrl,
        double const eb,
        int const    radius,
        T*& __restrict__ out_xdata,
        cudaStream_t const stream = nullptr)
    {
        derive_and_set_rt_len(interpreted_len3);

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
        dim3               interpreted_len3,
        T*                 in_anchor,
        E*                 in_errctrl,
        double const       eb,
        int const          radius,
        T*&                in_outlier__out_xdata,
        cudaStream_t const stream = nullptr)
    {
        derive_and_set_rt_len(interpreted_len3);

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