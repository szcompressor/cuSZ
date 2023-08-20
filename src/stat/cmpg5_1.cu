/**
 * @file cmpg4_2.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-03
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "detail/maxerr_thrust.inl"
#include "stat/compare_gpu.hh"

#define THRUSTGPU_ASSESS(Tliteral, T)           \
    template void psz::thrustgpu_get_maxerr<T>( \
        T * reconstructed, T * original, size_t len, T & maximum_val, size_t & maximum_loc, bool destructive);

THRUSTGPU_ASSESS(fp32, float);

#undef THRUSTGPU_ASSESS