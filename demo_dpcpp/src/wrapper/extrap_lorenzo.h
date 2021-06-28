#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
/**
 * @file extrap_lorenzo.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-06-16
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_WRAPPER_EXTRAP_LORENZO_H
#define CUSZ_WRAPPER_EXTRAP_LORENZO_H

template <typename Data, typename Quant, typename FP, int NDIM, int DATA_SUBSIZE>
void compress_lorenzo_construct(Data* data, Quant* quant, sycl::range<3> size3, FP eb, int radius);

template <typename Data, typename Quant, typename FP, int NDIM, int DATA_SUBSIZE>
void decompress_lorenzo_reconstruct(Data* data, Quant* quant, sycl::range<3> size3, FP eb, int radius);

#endif