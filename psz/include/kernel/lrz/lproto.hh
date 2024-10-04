/**
 * @file l21.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-01
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef D5965FDA_3E90_4AC4_A53B_8439817D7F1C
#define D5965FDA_3E90_4AC4_A53B_8439817D7F1C

#include <stdint.h>

#include "cusz/type.h"
#include "mem/compact.hh"
#include "port.hh"

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)

namespace psz::cuhip::proto {

template <typename T, typename E>
pszerror GPU_c_lorenzo_nd_with_outlier(
    T* const data, dim3 const len3, PROPER_EB const eb, int const radius,
    E* const eq, void* _outlier, float* time_elapsed, void* stream);

template <typename T, typename E>
pszerror GPU_x_lorenzo_nd(
    E* eq, dim3 const len3, T* outlier, PROPER_EB const eb, int const radius,
    T* xdata, float* time_elapsed, void* stream);

}  // namespace psz::cuhip::proto

#endif

#if defined(PSZ_USE_1API)

namespace psz::dpcpp::proto {
template <typename T, typename E>
pszerror GPU_c_lorenzo_nd_with_outlier(
    T* const data, sycl::range<3> const len3, PROPER_EB const eb,
    int const radius, E* const eq, void* _outlier, float* time_elapsed,
    void* stream);

template <typename T, typename E>
pszerror GPU_x_lorenzo_nd(
    E* eq, sycl::range<3> const len3, T* outlier, PROPER_EB const eb,
    int const radius, T* xdata, float* time_elapsed, void* stream);

}  // namespace psz::dpcpp::proto

#endif

#endif /* D5965FDA_3E90_4AC4_A53B_8439817D7F1C */
