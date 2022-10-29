/**
 * @file compare.h
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-09
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef CE05A256_23CB_4243_8839_B1FDA9C540D2
#define CE05A256_23CB_4243_8839_B1FDA9C540D2

#ifdef __cplus_plus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include "../cusz/type.h"

#define DESCRIPTION(Tliteral, T) void thrustgpu_get_extrema_rawptr_T##Tliteral(T* d_ptr, size_t len, T res[4]);

#define COMPARE_LOSSLESS(Tliteral, T)                                  \
    bool cppstd_identical_T##Tliteral(T* d1, T* d2, size_t const len); \
    bool thrustgpu_identical_T##Tliteral(T* d1, T* d2, size_t const len);

#define COMPARE_LOSSY(Tliteral, T)                                                                                     \
    bool cppstd_error_bounded_T##Tliteral(T* a, T* b, size_t const len, double const eb, size_t* first_faulty_idx);    \
    void cppstd_assess_quality_T##Tliteral(cusz_stats* s, T* xdata, T* odata, size_t const len);                       \
                                                                                                                       \
    bool thrustgpu_error_bounded_T##Tliteral(T* a, T* b, size_t const len, double const eb, size_t* first_faulty_idx); \
    void thrustgpu_assess_quality_T##Tliteral(cusz_stats* s, T* xdata, T* odata, size_t const len);

DESCRIPTION(ui8, uint8_t)
DESCRIPTION(ui16, uint16_t)
DESCRIPTION(ui32, uint32_t)
DESCRIPTION(fp32, float)
DESCRIPTION(fp64, double)

COMPARE_LOSSLESS(fp32, float)
COMPARE_LOSSLESS(fp64, double)
COMPARE_LOSSLESS(ui8, uint8_t)
COMPARE_LOSSLESS(ui16, uint16_t)
COMPARE_LOSSLESS(ui32, uint32_t)

COMPARE_LOSSY(fp32, float)
COMPARE_LOSSY(fp64, double)

#undef CPPSTD_COMPARE

#ifdef __cplus_plus
}
#endif

#endif /* CE05A256_23CB_4243_8839_B1FDA9C540D2 */
