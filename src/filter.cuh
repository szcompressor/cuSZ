// jtian 20-05-03

#ifndef FILTER_CUH
#define FILTER_CUH

#include <iostream>
#include "stdio.h"
using std::cout;
using std::endl;

namespace Prototype {

template <typename T, int DS, int tBLK>
__global__ void binning2d(T* input, T* output, size_t d0, size_t d1, size_t new_d0, size_t new_d1);

// template <typename T, int BLK>
//__host__ void binning2d(T* input, T* output, size_t dim0, size_t dim1, size_t new_dim0, size_t new_dim1, size_t xid, size_t yid);

}  // namespace Prototype

#endif
