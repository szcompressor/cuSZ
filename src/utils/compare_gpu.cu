/**
 * @file _compare.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-09
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "../detail/compare_gpu.inl"
#include "utils/compare.h"
#include "utils/compare_gpu.hh"

#define THRUSTGPU_DESCRIPTION(Tliteral, T)                                        \
    void thrustgpu_get_extrema_rawptr_T##Tliteral(T* d_ptr, size_t len, T res[4]) \
    {                                                                             \
        gpusz::detail::thrustgpu_get_extrema_rawptr<T>(d_ptr, len, res);          \
    }

#define THRUSTGPU_COMPARE_LOSSLESS(Tliteral, T)                          \
    bool thrustgpu_identical_T##Tliteral(T* d1, T* d2, size_t const len) \
    {                                                                    \
        return gpusz::detail::thrustgpu_identical<T>(d1, d2, len);       \
    }

#define THRUSTGPU_COMPARE_LOSSY(Tliteral, T)                                                       \
    bool thrustgpu_error_bounded_T##Tliteral(                                                      \
        T* a, T* b, size_t const len, double const eb, size_t* first_faulty_idx = nullptr)         \
    {                                                                                              \
        return gpusz::detail::thrustgpu_error_bounded<T>(a, b, len, eb, first_faulty_idx);         \
    }                                                                                              \
                                                                                                   \
    void thrustgpu_assess_quality_T##Tliteral(cusz_stats* s, T* xdata, T* odata, size_t const len) \
    {                                                                                              \
        gpusz::detail::thrustgpu_assess_quality<T>(s, xdata, odata, len);                          \
    }

THRUSTGPU_DESCRIPTION(ui8, uint8_t)
THRUSTGPU_DESCRIPTION(ui16, uint16_t)
THRUSTGPU_DESCRIPTION(ui32, uint32_t)
THRUSTGPU_DESCRIPTION(fp32, float)
THRUSTGPU_DESCRIPTION(fp64, double)

THRUSTGPU_COMPARE_LOSSLESS(fp32, float)
THRUSTGPU_COMPARE_LOSSLESS(fp64, double)
THRUSTGPU_COMPARE_LOSSLESS(ui8, uint8_t)
THRUSTGPU_COMPARE_LOSSLESS(ui16, uint16_t)
THRUSTGPU_COMPARE_LOSSLESS(ui32, uint32_t)

THRUSTGPU_COMPARE_LOSSY(fp32, float)
THRUSTGPU_COMPARE_LOSSY(fp64, double)

#undef THRUSTGPU_DESCRIPTION
#undef THRUSTGPU_COMPARE_LOSSLESS
#undef THRUSTGPU_COMPARE_LOSSY

#define THRUSTGPU_DESCRIPTION(Tliteral, T)                                   \
    template <>                                                              \
    void gpusz::thrustgpu_get_extrema_rawptr(T* d_ptr, size_t len, T res[4]) \
    {                                                                        \
        thrustgpu_get_extrema_rawptr_T##Tliteral(d_ptr, len, res);           \
    }

#define THRUSTGPU_COMPARE_LOSSLESS(Tliteral, T)                          \
    template <>                                                          \
    bool gpusz::thrustgpu_identical<T>(T * d1, T * d2, size_t const len) \
    {                                                                    \
        return thrustgpu_identical_T##Tliteral(d1, d2, len);             \
    }

#define THRUSTGPU_COMPARE_LOSSY(Tliteral, T)                                                                          \
    template <>                                                                                                       \
    bool gpusz::thrustgpu_error_bounded<T>(T * a, T * b, size_t const len, double const eb, size_t* first_faulty_idx) \
    {                                                                                                                 \
        return thrustgpu_error_bounded_T##Tliteral(a, b, len, eb, first_faulty_idx);                                  \
    }                                                                                                                 \
                                                                                                                      \
    template <>                                                                                                       \
    void gpusz::thrustgpu_assess_quality<T>(cusz_stats * s, T * xdata, T * odata, size_t const len)                   \
    {                                                                                                                 \
        thrustgpu_assess_quality_T##Tliteral(s, xdata, odata, len);                                                   \
    }

THRUSTGPU_DESCRIPTION(ui8, uint8_t)
THRUSTGPU_DESCRIPTION(ui16, uint16_t)
THRUSTGPU_DESCRIPTION(ui32, uint32_t)
THRUSTGPU_DESCRIPTION(fp32, float)
THRUSTGPU_DESCRIPTION(fp64, double)

THRUSTGPU_COMPARE_LOSSLESS(fp32, float)
THRUSTGPU_COMPARE_LOSSLESS(fp64, double)
THRUSTGPU_COMPARE_LOSSLESS(ui8, uint8_t)
THRUSTGPU_COMPARE_LOSSLESS(ui16, uint16_t)
THRUSTGPU_COMPARE_LOSSLESS(ui32, uint32_t)

THRUSTGPU_COMPARE_LOSSY(fp32, float);
THRUSTGPU_COMPARE_LOSSY(fp64, double);

#undef THRUSTGPU_DESCRIPTION
#undef THRUSTGPU_COMPARE_LOSSLESS
#undef THRUSTGPU_COMPARE_LOSSY
