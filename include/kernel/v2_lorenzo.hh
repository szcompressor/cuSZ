/**
 * @file v2_lorenzo.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-01-23
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef CD52BDA6_9376_43FF_BFDA_693204FA8762
#define CD52BDA6_9376_43FF_BFDA_693204FA8762

#include "compaction.hh"
#include "cusz/type.h"

template <typename T, typename E, typename FP>
cusz_error_status v2_compress_predict_lorenzo_i(
    T* const           data,          // input
    dim3 const         data_len3,     //
    double const       eb,            // input (config)
    int const          radius,        //
    E* const           eq,            // output
    dim3 const         eq_len3,       //
    T* const           anchor,        //
    dim3 const         anchor_len3,   //
    CompactCudaDram<T> outlier,       //
    float*             time_elapsed,  // optional
    cudaStream_t       stream);             //

#endif /* CD52BDA6_9376_43FF_BFDA_693204FA8762 */
