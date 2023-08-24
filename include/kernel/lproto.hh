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

#include <cuda_runtime.h>
#include <stdint.h>
#include "mem/compact.hh"
#include "cusz/type.h"

template <typename T, typename EQ = int32_t >
cusz_error_status psz_comp_lproto(
    T* const     data,          // input
    dim3 const   len3,          //
    double const eb,            // input (config)
    int const    radius,        //
    EQ* const    eq,            // output
    void*        _outlier,      //
    float*       time_elapsed,  // optional
    cudaStream_t stream);       //

template <typename T, typename EQ = int32_t>
cusz_error_status psz_decomp_lproto(
    EQ*          eq,                 // input
    dim3 const   len3,               //
    T*           scattered_outlier,  //
    double const eb,                 // input (config)
    int const    radius,             //
    T*           xdata,              // output
    float*       time_elapsed,       // optional
    cudaStream_t stream);

#endif /* D5965FDA_3E90_4AC4_A53B_8439817D7F1C */
