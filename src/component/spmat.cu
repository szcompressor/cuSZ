/**
 * @file spmat.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-28
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "component/spmat.cuh"

template struct cusz::SpcodecCSR<float, uint32_t>::impl;
