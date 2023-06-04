/**
 * @file cmp3g.cu
 * @author Jiannan Tian
 * @brief (split to speed up buid process; part 3)
 * @version 0.3
 * @date 2022-11-03
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "detail/equal_thrust.inl"
#include "stat/compare.h"
#include "stat/compare_gpu.hh"

#define THRUSTGPU_COMPARE_LOSSY(Tliteral, T)       \
    template bool psz::thrustgpu_error_bounded<T>( \
        T * a, T * b, size_t const len, double const eb, size_t* first_faulty_idx);

THRUSTGPU_COMPARE_LOSSY(fp32, float);
THRUSTGPU_COMPARE_LOSSY(fp64, double);

#undef THRUSTGPU_COMPARE_LOSSY
