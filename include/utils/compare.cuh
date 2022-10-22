/**
 * @file compare.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-09
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef B0EE0E82_B3AA_4946_A589_A3A6A83DD862
#define B0EE0E82_B3AA_4946_A589_A3A6A83DD862

#include "compare.h"

namespace gpusz {

template <typename T>
bool thrustgpu_identical(T* d1, T* d2, size_t const len);

template <typename T>
bool thrustgpu_error_bounded(T* a, T* b, size_t const len, double const eb, size_t* first_faulty_idx);

template <typename T>
void thrustgpu_assess_quality(cusz_stats* s, T* xdata, T* odata, size_t const len);

}  // namespace gpusz

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

THRUSTGPU_COMPARE_LOSSLESS(fp32, float)
THRUSTGPU_COMPARE_LOSSLESS(fp64, double)
THRUSTGPU_COMPARE_LOSSLESS(ui8, uint8_t)
THRUSTGPU_COMPARE_LOSSLESS(ui16, uint16_t)
THRUSTGPU_COMPARE_LOSSLESS(ui32, uint32_t)

THRUSTGPU_COMPARE_LOSSY(fp32, float);
THRUSTGPU_COMPARE_LOSSY(fp64, double);

#undef THRUSTGPU_COMPARE_LOSSLESS
#undef THRUSTGPU_COMPARE_LOSSY

#endif /* B0EE0E82_B3AA_4946_A589_A3A6A83DD862 */
