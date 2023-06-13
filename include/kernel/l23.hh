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

#include <cuda_runtime.h>
#include <stdint.h>
#include "compaction.hh"
#include "cusz/type.h"

template <typename T, typename EQ = int32_t, typename FP = T>
cusz_error_status psz_comp_l23(
    T* const     data,          // input
    dim3 const   len3,          //
    double const eb,            // input (config)
    int const    radius,        //
    EQ* const    eq,            // output
    T*           outlier,       //
    float*       time_elapsed,  // optional
    cudaStream_t stream);       //

template <typename T, typename EQ = int32_t, typename FP = T>
cusz_error_status psz_decomp_l23(
    EQ*          eq,            // input
    dim3 const   len3,          //
    T*           outlier,       //
    double const eb,            // input (config)
    int const    radius,        //
    T*           xdata,         // output
    float*       time_elapsed,  // optional
    cudaStream_t stream);

#endif /* B297267F_4731_48DB_8128_BBD202027EB7 */
