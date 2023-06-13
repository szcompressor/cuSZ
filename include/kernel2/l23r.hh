/**
 * @file l23r.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-04-05
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef EBA07D6C_FD5C_446C_9372_F78DDB5E2B34
#define EBA07D6C_FD5C_446C_9372_F78DDB5E2B34

#include <cuda_runtime.h>

#include <cstdint>

#include "../cusz/type.h"
#include "compaction.hh"

template <typename T>
void psz_adhoc_scttr(
    T* val, uint32_t* idx, int const n, T* out, float* milliseconds,
    cudaStream_t stream);

#define psz_comp_l23r compress_predict_lorenzo_i_rolling

template <typename T, bool UsePnEnc = false, typename Eq = uint32_t>
cusz_error_status compress_predict_lorenzo_i_rolling(
    T* const data, dim3 const len3, double const eb, int const radius,
    Eq* const eq, void* _outlier, float* time_elapsed, cudaStream_t stream);

#endif /* EBA07D6C_FD5C_446C_9372_F78DDB5E2B34 */
