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

#ifndef CUSZ_COMPONENT_INTERP_SPLINE_CUH
#define CUSZ_COMPONENT_INTERP_SPLINE_CUH

#include <exception>
#include <limits>
#include <numeric>

#include "../common.hh"
#include "../component/prediction.hh"
#include "../kernel/spline3.cuh"
#include "../utils.hh"

#define ALLOCDEV(VAR, SYM, NBYTE) \
    cudaMalloc(&d_##VAR, NBYTE);  \
    cudaMemset(d_##VAR, 0x0, NBYTE);

#define FREEDEV(VAR)       \
    if (d_##VAR) {         \
        cudaFree(d_##VAR); \
        d_##VAR = nullptr; \
    }

namespace cusz {

// override
// template <typename T, typename E, typename FP>
// size_t PredictorSpline3<T, E, FP>::impl::get_alloclen_quant() const
// {
//     auto m = Reinterpret1DTo2D::get_square_size(this->alloclen.assigned.quant);
//     return m * m;
// }

// template <typename T, typename E, typename FP>
// size_t PredictorSpline3<T, E, FP>::impl::get_len_quant() const
// {
//     auto m = Reinterpret1DTo2D::get_square_size(this->rtlen.assigned.quant);
//     return m * m;
// }

// template <typename T, typename E, typename FP>
// E* PredictorSpline3<T, E, FP>::impl::expose_quant() const
// {
//     return d_errctrl;
// }

// template <typename T, typename E, typename FP>
// E* PredictorSpline3<T, E, FP>::impl::expose_errctrl() const
// {
//     return d_errctrl;
// }

// template <typename T, typename E, typename FP>
// T* PredictorSpline3<T, E, FP>::impl::expose_anchor() const
// {
//     return d_anchor;
// }

// template <typename T, typename E, typename FP>
// T* PredictorSpline3<T, E, FP>::impl::expose_outlier() const
// {
//     return d_outlier;
// }

// template <typename T, typename E, typename FP>
// void PredictorSpline3<T, E, FP>::impl::derive_alloclen(dim3 base, cusz_predictortype)
// {
//     int sublen[3]      = {32, 8, 8};
//     int anchor_step[3] = {8, 8, 8};

//     this->__derive_len(base, this->alloclen, sublen, anchor_step, true);
// }

// template <typename T, typename E, typename FP>
// void PredictorSpline3<T, E, FP>::impl::derive_rtlen(dim3 base, cusz_predictortype)
// {
//     int sublen[3]      = {32, 8, 8};
//     int anchor_step[3] = {8, 8, 8};

//     this->__derive_len(base, this->rtlen, sublen, anchor_step, true);
// }

// template <typename T, typename E, typename FP>
// void PredictorSpline3<T, E, FP>::impl::init(dim3 _size, bool _dbg_mode)
// {
//     derive_alloclen(_size);
//     if (_dbg_mode) this->debug_list_alloclen<T, E, FP>(true);
//     init_continue();
// }

// template <typename T, typename E, typename FP>
// void PredictorSpline3<T, E, FP>::impl::init(size_t x, size_t y, size_t z, bool dbg_print)
// {
//     auto len3 = dim3(x, y, z);
//     init(len3, dbg_print);
// }

// /**
//  * @brief Allocate workspace according to the input size.
//  *
//  * @param xyz (host variable) 3D size for input data
//  * @param dbg_managed (host variable) use unified memory for debugging
//  * @param _delay_postquant_dummy (host variable) (future) control the delay of postquant
//  * @param _outlier_overlapped (host variable) (future) control the input-output overlapping
//  */
// template <typename T, typename E, typename FP>
// void PredictorSpline3<T, E, FP>::impl::init_continue(bool _delay_postquant_dummy, bool _outlier_overlapped)
// {
//     // config
//     delay_postquant_dummy = _delay_postquant_dummy;
//     outlier_overlapped    = _outlier_overlapped;

//     // allocate
//     auto nbyte_anchor = sizeof(T) * this->get_alloclen_anchor();
//     cudaMalloc(&d_anchor, nbyte_anchor);
//     cudaMemset(d_anchor, 0x0, nbyte_anchor);

//     auto nbyte_errctrl = sizeof(E) * this->get_alloclen_quant();
//     cudaMalloc(&d_errctrl, nbyte_errctrl);
//     cudaMemset(d_errctrl, 0x0, nbyte_errctrl);

//     if (not outlier_overlapped) {
//         auto nbyte_outlier = sizeof(T) * this->get_alloclen_quant();
//         cudaMalloc(&d_outlier, nbyte_outlier);
//         cudaMemset(d_outlier, 0x0, nbyte_outlier);
//     }
// }

// template <typename T, typename E, typename FP>
// PredictorSpline3<T, E, FP>::impl::impl()
// {
// }

// template <typename T, typename E, typename FP>
// PredictorSpline3<T, E, FP>::impl::~impl()
// {
//     FREEDEV(anchor);
//     FREEDEV(errctrl);
// }

// /**
//  * @brief clear GPU buffer, may affect performance; essentially for debugging
//  *
//  */
// template <typename T, typename E, typename FP>
// void PredictorSpline3<T, E, FP>::impl::clear_buffer()
// {
//     cudaMemset(d_anchor, 0x0, sizeof(T) * this->get_len_anchor());
//     cudaMemset(d_errctrl, 0x0, sizeof(E) * get_len_quant());
// }

// /**
//  * @brief Construct error-control code & outlier; input and outlier overlap each other. Thus, it's destructive.
//  *
//  * @param in_data (device array) input data and output outlier
//  * @param cfg_eb (host variable) error bound; configuration
//  * @param cfg_radius (host variable) radius to control the bound; configuration
//  * @param ptr_anchor (device array) output anchor point
//  * @param ptr_errctrl (device array) output error-control code; if range-limited integer, it is quant-code
//  * @param stream CUDA stream
//  */
// template <typename T, typename E, typename FP>
// void PredictorSpline3<T, E, FP>::impl::construct(
//     dim3 const   len3,
//     TITER        in_data,
//     TITER&       out_anchor,
//     EITER&       out_errctrl,
//     double const eb,
//     int const    radius,
//     cudaStream_t stream)
// {
//     derive_rtlen(len3);  // placeholder
//     this->check_rtlen();

//     this->debug_list_rtlen<T, E, FP>(true);

//     auto ebx2 = eb * 2;
//     auto eb_r = 1 / eb;

//     out_anchor  = d_anchor;
//     out_errctrl = d_errctrl;

//     if (dbg_mode) {
//         printf("\nSpline3::construct dbg:\n");
//         printf("ebx2: %lf\n", ebx2);
//         printf("eb_r: %lf\n", eb_r);
//     }

//     cuda_timer_t timer;
//     timer.timer_start();

//     cusz::c_spline3d_infprecis_32x8x8data<TITER, EITER, float, 256, false>
//         <<<this->rtlen.nblock, dim3(256, 1, 1), 0, stream>>>             //
//         (in_data, this->rtlen.base.len3, this->rtlen.base.leap,          //
//          d_errctrl, this->rtlen.aligned.len3, this->rtlen.aligned.leap,  //
//          d_anchor, this->rtlen.anchor.leap,                              //
//          eb_r, ebx2, radius);

//     timer.timer_end();

//     if (stream)
//         CHECK_CUDA(cudaStreamSynchronize(stream));
//     else
//         CHECK_CUDA(cudaDeviceSynchronize());

//     this->time_elapsed = timer.get_time_elapsed();
// }

// /**
//  * @brief Reconstruct data from error-control code & outlier; outlier and output overlap each other; destructive for
//  * outlier.
//  *
//  * @param in_anchor (device array) input anchor
//  * @param in_errctrl (device array) input error-control code
//  * @param cfg_eb (host variable) error bound; configuration
//  * @param cfg_radius (host variable) radius to control the bound; configuration
//  * @param in_outlier__out_xdata (device array) output reconstructed data, overlapped with input outlier
//  * @param stream CUDA stream
//  */
// template <typename T, typename E, typename FP>
// void PredictorSpline3<T, E, FP>::impl::reconstruct(
//     dim3         len3,
//     TITER&       in_outlier__out_xdata,
//     TITER        in_anchor,
//     EITER        in_errctrl,
//     double const cfg_eb,
//     int const    cfg_radius,
//     cudaStream_t stream)
// {
//     derive_rtlen(len3);
//     this->check_rtlen();

//     auto ebx2 = cfg_eb * 2;
//     auto eb_r = 1 / cfg_eb;

//     cuda_timer_t timer;
//     timer.timer_start();

//     cusz::x_spline3d_infprecis_32x8x8data<EITER, TITER, float, 256>
//         <<<this->rtlen.nblock, dim3(256, 1, 1), 0, stream>>>                   //
//         (in_errctrl, this->rtlen.aligned.len3, this->rtlen.aligned.leap,       //
//          in_anchor, this->rtlen.anchor.len3, this->rtlen.anchor.leap,          //
//          in_outlier__out_xdata, this->rtlen.base.len3, this->rtlen.base.leap,  //
//          eb_r, ebx2, cfg_radius);

//     timer.timer_end();

//     if (stream)
//         CHECK_CUDA(cudaStreamSynchronize(stream));
//     else
//         CHECK_CUDA(cudaDeviceSynchronize());

//     this->time_elapsed = timer.get_time_elapsed();
// }

// template <typename T, typename E, typename FP>
// float PredictorSpline3<T, E, FP>::impl::get_time_elapsed() const
// {
//     return time_elapsed;
// }

}  // namespace cusz

#undef FREEDEV
#undef ALLOCDEV

#endif
