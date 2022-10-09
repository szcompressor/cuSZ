/**
 * @file compare.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-09
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef C93C3857_8821_4988_B6F0_4E885060F642
#define C93C3857_8821_4988_B6F0_4E885060F642

#include "compare.h"

namespace gpusz {

template <typename T>
bool cppstd_identical(T* d1, T* d2, size_t const len);

template <typename T>
bool cppstd_error_bounded(T* a, T* b, size_t const len, double const eb, size_t* first_faulty_idx);

template <typename T>
void cppstd_assess_quality(cusz_stats* s, T* xdata, T* odata, size_t const len);

}  // namespace gpusz

#define CPPSTD_COMPARE_LOSSLESS(Tliteral, T)                          \
    template <>                                                       \
    bool gpusz::cppstd_identical<T>(T * d1, T * d2, size_t const len) \
    {                                                                 \
        return cppstd_identical_T##Tliteral(d1, d2, len);             \
    }

#define CPPSTD_COMPARE_LOSSY(Tliteral, T)                                                                          \
    template <>                                                                                                    \
    bool gpusz::cppstd_error_bounded<T>(T * a, T * b, size_t const len, double const eb, size_t* first_faulty_idx) \
    {                                                                                                              \
        return cppstd_error_bounded_T##Tliteral(a, b, len, eb, first_faulty_idx);                                  \
    }                                                                                                              \
                                                                                                                   \
    template <>                                                                                                    \
    void gpusz::cppstd_assess_quality<T>(cusz_stats * s, T * xdata, T * odata, size_t const len)                   \
    {                                                                                                              \
        cppstd_assess_quality_T##Tliteral(s, xdata, odata, len);                                                   \
    }

CPPSTD_COMPARE_LOSSLESS(fp32, float)
CPPSTD_COMPARE_LOSSLESS(fp64, double)
CPPSTD_COMPARE_LOSSLESS(ui8, uint8_t)
CPPSTD_COMPARE_LOSSLESS(ui16, uint16_t)
CPPSTD_COMPARE_LOSSLESS(ui32, uint32_t)

CPPSTD_COMPARE_LOSSY(fp32, float);
CPPSTD_COMPARE_LOSSY(fp64, double);

#undef CPPSTD_COMPARE_LOSSLESS
#undef CPPSTD_COMPARE_LOSSY

#endif /* C93C3857_8821_4988_B6F0_4E885060F642 */
