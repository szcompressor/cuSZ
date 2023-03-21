/**
 * @file cmp2g.cu
 * @author Jiannan Tian
 * @brief (split to speed up buid process; part 2)
 * @version 0.3
 * @date 2022-11-03
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "../detail/compare_gpu.inl"
#include "stat/compare.h"
#include "stat/compare_gpu.hh"

#define THRUSTGPU_COMPARE_LOSSLESS(Tliteral, T)                          \
    bool thrustgpu_identical_T##Tliteral(T* d1, T* d2, size_t const len) \
    {                                                                    \
        return psz::detail::thrustgpu_identical<T>(d1, d2, len);         \
    }                                                                    \
                                                                         \
    template <>                                                          \
    bool psz::thrustgpu_identical<T>(T * d1, T * d2, size_t const len)   \
    {                                                                    \
        return thrustgpu_identical_T##Tliteral(d1, d2, len);             \
    }

THRUSTGPU_COMPARE_LOSSLESS(fp32, float)
THRUSTGPU_COMPARE_LOSSLESS(fp64, double)
THRUSTGPU_COMPARE_LOSSLESS(ui8, uint8_t)
THRUSTGPU_COMPARE_LOSSLESS(ui16, uint16_t)
THRUSTGPU_COMPARE_LOSSLESS(ui32, uint32_t)

#undef THRUSTGPU_COMPARE_LOSSLESS
