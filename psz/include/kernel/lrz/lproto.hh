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
#include "kernel/lrz/lproto.hh"
#include "mem/compact.hh"

namespace psz::cuhip::proto {

template <typename T, typename EQ = int32_t>
pszerror GPU_c_lorenzo_nd_with_outlier(
    T* const data,  // input
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    dim3 const len3,
#elif defined(PSZ_USE_1API)
    sycl::range<3> const len3,
#endif
    double const eb,      // input (config)
    int const radius,     //
    EQ* const eq,         // output
    void* _outlier,       //
    float* time_elapsed,  // optional
    void* stream);        //

template <typename T, typename EQ = int32_t>
pszerror GPU_x_lorenzo_nd(
    EQ* eq,  // input
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    dim3 const len3,
#elif defined(PSZ_USE_1API)
    sycl::range<3> const len3,
#endif
    T* scattered_outlier,  //
    double const eb,       // input (config)
    int const radius,      //
    T* xdata,              // output
    float* time_elapsed,   // optional
    void* stream);

}  // namespace psz::cuhip::proto

#endif /* D5965FDA_3E90_4AC4_A53B_8439817D7F1C */
