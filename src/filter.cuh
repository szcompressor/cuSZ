/**
 * @file filter.cuh
 * @author Jiannan Tian
 * @brief Filters for preprocessing of cuSZ (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-05-03
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
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

template <typename Data, int DownscaleFactor, int tBLK>
__global__ void binning2d(Data*, Data*, size_t, size_t, size_t, size_t);

}  // namespace Prototype

#endif
