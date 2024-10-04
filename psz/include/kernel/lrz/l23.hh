/**
 * @file l23.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-01
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef B297267F_4731_48DB_8128_BBD202027EB7
#define B297267F_4731_48DB_8128_BBD202027EB7

#include <stdint.h>

#include "cusz/type.h"
#include "mem/compact.hh"
#include "port.hh"

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)

namespace psz::cuhip {

template <typename T, typename E, psz_timing_mode TIMING = SYNC_BY_STREAM>
pszerror GPU_x_lorenzo_nd(
    E* eq, dim3 const len3, T* outlier, PROPER_EB const eb, int const radius,
    T* xdata, float* time_elapsed, void* stream);

}

#endif

#if defined(PSZ_USE_1API)

namespace psz::dpcpp {

template <typename T, typename E, psz_timing_mode TIMING = SYNC_BY_STREAM>
pszerror GPU_x_lorenzo_nd(
    E* eq, sycl::range<3> const len3, T* outlier, PROPER_EB const eb,
    int const radius, T* xdata, float* time_elapsed, void* stream);

}

#endif

#endif /* B297267F_4731_48DB_8128_BBD202027EB7 */
