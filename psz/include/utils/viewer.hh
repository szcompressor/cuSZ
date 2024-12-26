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
#include "mem/cxx_array.h"
#include "mem/cxx_memobj.h"

using std::string;
using std::vector;

template <typename T>
using memobj = _portable::memobj<T>;

template <typename T>
using array3 = _portable::array3<T>;

// deps
namespace psz::utils {

template <typename T>
void print_metrics_cross(psz_statistics* s, size_t comp_bytes = 0, bool gpu_checker = false);

void print_metrics_auto(double* lag1_cor, double* lag2_cor);

template <typename T>
void view(psz_header* header, memobj<T>* xdata, memobj<T>* cmp, string const& compare);

}  // namespace psz::utils

// TODO have not passed test
template <typename T, psz_runtime P = CUDA>
pszerror pszcxx_evaluate_quality_gpu(array3<T> xdata, array3<T> odata);

template <typename T, psz_runtime P = CUDA>
void pszcxx_evaluate_quality_gpu(T* xdata, T* odata, size_t len, size_t comp_bytes = 0);

template <typename T>
void pszcxx_evaluate_quality_cpu(
    T* _d1, T* _d2, size_t len, size_t comp_bytes = 0, bool from_device = true);

#endif /* C6EF99AE_F0D7_485B_ADE4_8F55666CA96C */
