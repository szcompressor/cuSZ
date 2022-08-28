/**
 * @file spvec.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-03-01
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "detail/spvec.cuh"

template struct cusz::SpcodecVec<float>::impl;
template struct cusz::SpcodecVec<uint8_t>::impl;
template struct cusz::SpcodecVec<uint16_t>::impl;
template struct cusz::SpcodecVec<uint32_t>::impl;
// template struct cusz::SpcodecVec<double>::impl;
