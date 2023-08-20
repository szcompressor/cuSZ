/**
 * @file cmpg1_4.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-03
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "detail/extrema_thrust.inl"
#include "stat/compare_thrust.hh"

#define THRUSTGPU_DESCRIPTION(Tliteral, T) \
    template void psz::thrustgpu_get_extrema_rawptr(T* d_ptr, size_t len, T res[4]);

THRUSTGPU_DESCRIPTION(fp32, float)

#undef THRUSTGPU_DESCRIPTION