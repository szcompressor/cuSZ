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
#include "mem/cxx_sp_gpu.h"
#include "port.hh"

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)

namespace psz::cuhip {

template <typename T, typename Eq>
pszerror GPU_PROTO_c_lorenzo_nd_with_outlier(
    T* const in_data, dim3 const data_len3, Eq* const out_eq, void* out_outlier,
    PROPER_EB const eb, uint16_t const radius, float* time_elapsed, void* stream);

template <typename T, typename Eq>
pszerror GPU_PROTO_x_lorenzo_nd(
    Eq* in_eq, T* in_outlier, T* out_data, dim3 const data_len3, PROPER_EB const eb,
    int const radius, float* time_elapsed, void* stream);

}  // namespace psz::cuhip

#endif

#if defined(PSZ_USE_1API)

namespace psz::dpcpp::proto {
template <typename T, typename E>
pszerror GPU_c_lorenzo_nd_with_outlier(
    T* const data, sycl::range<3> const len3, PROPER_EB const eb, int const radius, E* const eq,
    void* _outlier, float* time_elapsed, void* stream);

template <typename T, typename E>
pszerror GPU_x_lorenzo_nd(
    E* eq, sycl::range<3> const len3, T* outlier, PROPER_EB const eb, int const radius, T* xdata,
    float* time_elapsed, void* stream);

}  // namespace psz::dpcpp::proto

#endif

#endif /* D5965FDA_3E90_4AC4_A53B_8439817D7F1C */
