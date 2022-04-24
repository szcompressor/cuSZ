/**
 * @file interp_spline3.cu
 * @author Jiannan Tian
 * @brief A high-level Spline3D wrapper. Allocations are explicitly out of called functions.
 * @version 0.3
 * @date 2021-06-15
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "component/interp_spline3.cuh"

template struct cusz::api::PredictorSpline3<float, float, float>::impl;
