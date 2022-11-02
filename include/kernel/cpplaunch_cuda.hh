/**
 * @file cpplaunch_cuda.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-07-27
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef COMPONENT_CALL_KERNEL_HH
#define COMPONENT_CALL_KERNEL_HH

#include "../cusz/type.h"
#include "../hf/hf_struct.h"

namespace cusz {

// 22-10-27 revise later
template <typename T, typename E, typename FP>
cusz_error_status cpplaunch_construct_Spline3(
    bool         NO_R_SEPARATE,
    T*           data,
    dim3 const   len3,
    T*           anchor,
    dim3 const   an_len3,
    E*           eq,
    dim3 const   ec_len3,
    double const eb,
    int const    radius,
    float*       time_elapsed,
    cudaStream_t stream);

// 22-10-27 revise later
template <typename T, typename E, typename FP>
cusz_error_status cpplaunch_reconstruct_Spline3(
    T*           xdata,
    dim3 const   len3,
    T*           anchor,
    dim3 const   an_len3,
    E*           eq,
    dim3 const   ec_len3,
    double const eb,
    int const    radius,
    float*       time_elapsed,
    cudaStream_t stream);

}  // namespace cusz

#endif
