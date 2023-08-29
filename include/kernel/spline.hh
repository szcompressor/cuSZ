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

#include "mem/memseg_cxx.hh"

typedef struct Spline3Len3 {
  // uint32_t blockx, blocky, blockz;
  uint32_t x_32, y_8, z_8;

  uint32_t len;
  uint32_t len_aligned;
  uint32_t len_anchor;

  dim3 grid_dim;

  dim3 l3_data, stride3_data;
  dim3 l3_aligned, stride3_aligned;
  dim3 l3_anchor, stride3_anchor;
} Spline3Len3;

void spline3_calc_sizes(void* _l3, Spline3Len3* s3l3);

template <typename T, typename E, typename FP = T>
int spline_construct(
    pszmem_cxx<T>* data, pszmem_cxx<T>* anchor, pszmem_cxx<E>* errctrl,
    void* _outlier, double eb, uint32_t radius, float* time, void* stream);

template <typename T, typename E, typename FP = T>
int spline_reconstruct(
    pszmem_cxx<T>* anchor, pszmem_cxx<E>* errctrl, pszmem_cxx<T>* xdata,
    double eb, uint32_t radius, float* time, void* stream);

#endif /* AA9BE6AD_ECA4_4267_A97F_B12C25A2B0C1 */
