/**
 * @file cmpg4_1.cu
 * @author Jiannan Tian
 * @brief (split to speed up buid process; part 4)
 * @version 0.3
 * @date 2022-11-03
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "../detail/compare_gpu.inl"
#include "stat/compare.h"
#include "stat/compare_gpu.hh"

#define THRUSTGPU_ASSESS(Tliteral, T)                                                              \
    void thrustgpu_assess_quality_T##Tliteral(cusz_stats* s, T* xdata, T* odata, size_t const len) \
    {                                                                                              \
        gpusz::detail::thrustgpu_assess_quality<T>(s, xdata, odata, len);                          \
    }

THRUSTGPU_ASSESS(fp32, float);

#undef THRUSTGPU_ASSESS
