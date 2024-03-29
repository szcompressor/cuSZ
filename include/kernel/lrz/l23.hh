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

// template <typename T, typename EQ = int32_t, bool TIMING = true>
// pszerror pszcxx_predict_lorenzo_unused(
//     T* const data,  // input
// #if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
//     dim3 const len3,
// #elif defined(PSZ_USE_1API)
//     sycl::range<3> const len3,
// #endif
//     PROPER_EB const eb,   // input (config)
//     int const radius,     //
//     EQ* const eq,         // output
//     T* outlier,           //
//     float* time_elapsed,  // optional
//     void* stream);        //

template <
    typename T,
    psz_timing_mode TIMING = CPU_BARRIER_AND_TIMING>
pszerror pszcxx_reverse_predict_lorenzo__internal(
    u4* eq,  // input
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    dim3 const len3,
#elif defined(PSZ_USE_1API)
    sycl::range<3> const len3,
#endif
    T* outlier,           //
    PROPER_EB const eb,   // input (config)
    int const radius,     //
    T* xdata,             // output
    float* time_elapsed,  // optional
    void* stream);

#endif /* B297267F_4731_48DB_8128_BBD202027EB7 */
