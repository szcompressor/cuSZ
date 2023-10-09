/**
 * @file l21var.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-01
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef AED3F205_ABA4_45F9_AC73_EF06654F6F3C
#define AED3F205_ABA4_45F9_AC73_EF06654F6F3C

#include <cuda_runtime.h>
#include <stdint.h>
#include "mem/compact.hh"
#include "cusz/type.h"

template <typename T, typename DeltaT, typename FP>
psz_error_status psz_comp_l21var(
    T*           data,
    dim3 const   len3,
    double const eb,
    DeltaT*      delta,
    bool*        signum,
    float*       time_elapsed,
    cudaStream_t stream);

template <typename T, typename DeltaT, typename FP>
psz_error_status psz_decomp_l21var(
    DeltaT*      delta,
    bool*        signum,
    dim3 const   len3,
    double const eb,
    T*           xdata,
    float*       time_elapsed,
    cudaStream_t stream);

#endif /* AED3F205_ABA4_45F9_AC73_EF06654F6F3C */
