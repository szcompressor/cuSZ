/**
 * @file cmpg1.cu
 * @author Jiannan Tian
 * @brief (split to speed up buid process; part 1)
 * @version 0.3
 * @date 2022-10-09
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
        parsz::detail::thrustgpu_get_extrema_rawptr<T>(d_ptr, len, res);          \
    }                                                                             \
                                                                                  \
    template <>                                                                   \
    void parsz::thrustgpu_get_extrema_rawptr(T* d_ptr, size_t len, T res[4])      \
    {                                                                             \
        thrustgpu_get_extrema_rawptr_T##Tliteral(d_ptr, len, res);                \
    }

THRUSTGPU_DESCRIPTION(ui8, uint8_t)

#undef THRUSTGPU_DESCRIPTION
