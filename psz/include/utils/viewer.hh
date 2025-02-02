/**
 * @file viewer.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-09
 * @deprecated 0.3.2
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef C6EF99AE_F0D7_485B_ADE4_8F55666CA96C
#define C6EF99AE_F0D7_485B_ADE4_8F55666CA96C

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <string>
#include <vector>

#include "cusz/type.h"
#include "header.h"

using std::string;
using std::vector;

namespace psz::analysis {

template <typename T>
void print_metrics_cross(psz_statistics* s, size_t comp_bytes = 0, bool gpu_checker = false);

void print_metrics_auto(double* lag1_cor, double* lag2_cor);

}  // namespace psz::analysis

namespace psz::analysis {

template <typename T, psz_runtime P = CUDA>
void GPU_evaluate_quality_and_print(T* xdata, T* odata, size_t len, size_t comp_bytes = 0);

template <typename T>
void CPU_evaluate_quality_and_print(
    T* _d1, T* _d2, size_t len, size_t comp_bytes = 0, bool from_device = true);

}  // namespace psz::analysis

#endif /* C6EF99AE_F0D7_485B_ADE4_8F55666CA96C */
