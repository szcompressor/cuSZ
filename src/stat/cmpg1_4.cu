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

#include "../detail/compare_gpu.inl"
#include "stat/compare.h"
#include "stat/compare_gpu.hh"

#define THRUSTGPU_DESCRIPTION(Tliteral, T)                                        \
    void thrustgpu_get_extrema_rawptr_T##Tliteral(T* d_ptr, size_t len, T res[4]) \
    {                                                                             \
        psz::detail::thrustgpu_get_extrema_rawptr<T>(d_ptr, len, res);            \
    }                                                                             \
                                                                                  \
    template <>                                                                   \
    void psz::thrustgpu_get_extrema_rawptr(T* d_ptr, size_t len, T res[4])        \
    {                                                                             \
        thrustgpu_get_extrema_rawptr_T##Tliteral(d_ptr, len, res);                \
    }

THRUSTGPU_DESCRIPTION(fp32, float)

#undef THRUSTGPU_DESCRIPTION