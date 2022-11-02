/**
 * @file _compare.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-09
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "../detail/compare_cpu.inl"
#include "stat/compare.h"

#define CPPSTD_COMPARE_LOSSLESS(Tliteral, T)                          \
    bool cppstd_identical_T##Tliteral(T* d1, T* d2, size_t const len) \
    {                                                                 \
        return gpusz::detail::cppstd_identical<T>(d1, d2, len);       \
    }

#define CPPSTD_COMPARE_LOSSY(Tliteral, T)                                                       \
    bool cppstd_error_bounded_T##Tliteral(                                                      \
        T* a, T* b, size_t const len, double const eb, size_t* first_faulty_idx = nullptr)      \
    {                                                                                           \
        return gpusz::detail::cppstd_error_bounded<T>(a, b, len, eb, first_faulty_idx);         \
    }                                                                                           \
                                                                                                \
    void cppstd_assess_quality_T##Tliteral(cusz_stats* s, T* xdata, T* odata, size_t const len) \
    {                                                                                           \
        gpusz::detail::cppstd_assess_quality<T>(s, xdata, odata, len);                          \
    }

CPPSTD_COMPARE_LOSSLESS(fp32, float)
CPPSTD_COMPARE_LOSSLESS(fp64, double)
CPPSTD_COMPARE_LOSSLESS(ui8, uint8_t)
CPPSTD_COMPARE_LOSSLESS(ui16, uint16_t)
CPPSTD_COMPARE_LOSSLESS(ui32, uint32_t)

CPPSTD_COMPARE_LOSSY(fp32, float)
CPPSTD_COMPARE_LOSSY(fp64, double)

#undef CPPSTD_COMPARE_LOSSLESS
#undef CPPSTD_COMPARE_LOSSY
