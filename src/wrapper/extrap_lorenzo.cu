/**
 * @file extrap_lorenzo.cu
 * @author Jiannan Tian
 * @brief A high-level LorenzoND wrapper. Allocations are explicitly out of called functions.
 * @version 0.3
 * @date 2021-06-16
 * (rev.1) 2021-09-18 (rev.2) 2022-01-10
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "extrap_lorenzo.cuh"

template class cusz::PredictorLorenzo<float, uint16_t, float>;
