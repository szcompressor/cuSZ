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

#include <cstdint>

#include "cusz/type.h"
#include "mem/compact.hh"

template <typename T, typename Eq = uint32_t, bool ZigZag = false>
cusz_error_status psz_comp_l23r(
    T* const data, dim3 const len3, double const eb, int const radius,
    Eq* const eq, void* _outlier, float* time_elapsed, void* _stream);

#endif /* EBA07D6C_FD5C_446C_9372_F78DDB5E2B34 */
