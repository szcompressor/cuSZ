/**
 * @file spline.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-08-03
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef AA9BE6AD_ECA4_4267_A97F_B12C25A2B0C1
#define AA9BE6AD_ECA4_4267_A97F_B12C25A2B0C1

#include <cstddef>
#include <cstdint>

#include "mem/memobj.hh"
using namespace portable;

template <typename T, typename E, typename FP = T>
int pszcxx_predict_spline(
    memobj<T>* data, memobj<T>* anchor, memobj<E>* errctrl,
    void* _outlier, double eb, uint32_t radius, float* time, void* stream);

template <typename T, typename E, typename FP = T>
int pszcxx_reverse_predict_spline(
    memobj<T>* anchor, memobj<E>* errctrl, memobj<T>* xdata,
    double eb, uint32_t radius, float* time, void* stream);

#endif /* AA9BE6AD_ECA4_4267_A97F_B12C25A2B0C1 */
