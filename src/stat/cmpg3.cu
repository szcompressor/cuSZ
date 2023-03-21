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

#include "../detail/compare_gpu.inl"
#include "stat/compare.h"
#include "stat/compare_gpu.hh"

#define THRUSTGPU_COMPARE_LOSSY(Tliteral, T)                                                                        \
    bool thrustgpu_error_bounded_T##Tliteral(                                                                       \
        T* a, T* b, size_t const len, double const eb, size_t* first_faulty_idx = nullptr)                          \
    {                                                                                                               \
        return psz::detail::thrustgpu_error_bounded<T>(a, b, len, eb, first_faulty_idx);                            \
    }                                                                                                               \
                                                                                                                    \
    template <>                                                                                                     \
    bool psz::thrustgpu_error_bounded<T>(T * a, T * b, size_t const len, double const eb, size_t* first_faulty_idx) \
    {                                                                                                               \
        return thrustgpu_error_bounded_T##Tliteral(a, b, len, eb, first_faulty_idx);                                \
    }

THRUSTGPU_COMPARE_LOSSY(fp32, float);
THRUSTGPU_COMPARE_LOSSY(fp64, double);

#undef THRUSTGPU_COMPARE_LOSSY
