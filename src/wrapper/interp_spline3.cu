/**
 * @file interp_spline3.cu
 * @author Jiannan Tian
 * @brief A high-level Spline3D wrapper. Allocations are explicitly out of called functions.
 * @version 0.3
 * @date 2021-06-15
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include <limits>
#include <numeric>
#include "../common.hh"
#include "../kernel/spline3.cuh"
#include "../utils.hh"
#include "interp_spline3.cuh"

namespace {

#ifndef __CUDACC__
struct __dim3_compat {
    unsigned int x, y, z;
    __dim3_compat(unsigned int _x, unsigned int _y, unsigned int _z) : x(_x), y(_y), z(_z){};
};

using dim3 = __size_compat;
#endif

}  // namespace

#define SPLINE3 cusz::Spline3<T, E, FP>

template <typename T, typename E, typename FP>
void SPLINE3::allocate_workspace(dim3 xyz, bool _delay_postquant_dummy, bool _outlier_overlapped)
{
    dimx = size.x;
    dimy = size.y;
    dimz = size.z;

    delay_postquant_dummy = _delay_postquant_dummy;
    outlier_overlapped    = _outlier_overlapped;

    // data size
    nblockx      = ConfigHelper::get_npart(dimx, BLOCK * 4);
    nblocky      = ConfigHelper::get_npart(dimy, BLOCK);
    nblockz      = ConfigHelper::get_npart(dimz, BLOCK);
    dimx_aligned = nblockx * 32;  // 235 -> 256
    dimy_aligned = nblocky * 8;   // 449 -> 456
    dimz_aligned = nblockz * 8;   // 449 -> 456
    len_aligned  = dimx_aligned * dimy_aligned * dimz_aligned;

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
#define ALLOCDEV(VAR, SYM, NBYTE) \
    cudaMalloc(&d_##VAR, NBYTE);  \
    cudaMemset(d_##VAR, 0x0, NBYTE);
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
#undef ALLCDEV
}

template <typename T, typename E, typename FP>
void SPLINE3::construct(
    TITER        in_data,
    double const cfg_eb,
    int const    cfg_radius,
    TITER&       out_anchor,
    EITER&       out_errctrl,
    cudaStream_t stream,
    T* const __restrict__ non_overlap_in_outlier)
{
    auto ebx2 = eb * 2;
    auto eb_r = 1 / eb;

    out_anchor  = d_anchor;
    out_errctrl = d_errctrl;

    cuda_timer_t timer;
    timer.timer_start();

    cusz::c_spline3d_infprecis_32x8x8data<TITER, EITER, float, 256, false>
        <<<dim3(nblockx, nblocky, nblockz), dim3(256, 1, 1), 0, stream>>>  //
        (in_data, size, leap,                                              //
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

template <typename T, typename E, typename FP>
void SPLINE3::reconstruct(
    TITER        in_anchor,
    EITER        in_errctrl,
    double const cfg_eb,
    int const    cfg_radius,
    TITER&       out_xdata,
    cudaStream_t stream,
    T* const __restrict__ non_overlap_in_outlier)
{
    auto ebx2 = cfg_eb * 2;
    auto eb_r = 1 / cfg_eb;

    cuda_timer_t timer;
    timer.timer_start();

    cusz::x_spline3d_infprecis_32x8x8data<EITER, TITER, float, 256>
        <<<dim3(nblockx, nblocky, nblockz), dim3(256, 1, 1), 0, stream>>>  //
        (in_errctrl, size_aligned, leap_aligned,                           //
         in_anchor, anchor_size, anchor_leap,                              //
         out_xdata, size, leap,                                            //
         eb_r, ebx2, radius);

    timer.timer_end();

    if (stream)
        CHECK_CUDA(cudaStreamSynchronize(stream));
    else
        CHECK_CUDA(cudaDeviceSynchronize());

    time_elapsed = timer.get_time_elapsed();
}

// template class cusz::Spline3<float, unsigned short, float>;
// template class cusz::Spline3<float, unsigned int, float>;
template class cusz::Spline3<float, float, float>;
