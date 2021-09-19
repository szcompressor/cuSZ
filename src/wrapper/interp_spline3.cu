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
#include "../kernel/spline3.cuh"
#include "../common.hh"
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
SPLINE3::Spline3(dim3 xyz, double _eb, int _radius, bool _delay_postquant_dummy)
{
    eb     = _eb;
    ebx2   = eb * 2;
    eb_r   = 1 / eb;
    radius = _radius;

    dimx = xyz.x;
    dimy = xyz.y;
    dimz = xyz.z;

    delay_postquant_dummy = _delay_postquant_dummy;

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
}

template <typename T, typename E, typename FP>
void SPLINE3::construct(TITER data, TITER anchor, EITER errctrl)
{
    cusz::c_spline3d_infprecis_32x8x8data<TITER, EITER, float, 256, false>
        <<<dim3(nblockx, nblocky, nblockz), dim3(256, 1, 1)>>>  //
        (data, size, leap,                                      //
         errctrl, size_aligned, leap_aligned,                   //
         anchor, anchor_leap,                                   //
         eb_r, ebx2, radius);
    cudaDeviceSynchronize();
}

template <typename T, typename E, typename FP>
void SPLINE3::reconstruct(TITER anchor, EITER errctrl, TITER xdata)
{
    cusz::x_spline3d_infprecis_32x8x8data<EITER, TITER, float, 256>
        <<<dim3(nblockx, nblocky, nblockz), dim3(256, 1, 1)>>>  //
        (errctrl, size_aligned, leap_aligned,                   //
         anchor, anchor_leap,                                   //
         xdata, size, leap,                                     //
         eb_r, ebx2, radius);
    cudaDeviceSynchronize();
}

template class cusz::Spline3<float, unsigned short, float>;
template class cusz::Spline3<float, unsigned int, float>;
