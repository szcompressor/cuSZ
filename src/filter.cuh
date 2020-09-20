/**
 * @file filter.cuh
 * @author Jiannan Tian
 * @brief Filters for preprocessing of cuSZ (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-05-03
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef FILTER_CUH
#define FILTER_CUH

#include <iostream>
#include "stdio.h"
using std::cout;
using std::endl;

namespace Prototype {

template <typename T, int DS, int tBLK>
__global__ void binning2d(T* input, T* output, size_t d0, size_t d1, size_t new_d0, size_t new_d1);

}  // namespace Prototype

#endif
