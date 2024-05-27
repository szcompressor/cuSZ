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

#include "cusz/suint.hh"
#include "cusz/type.h"
#include "port.hh"

template <typename T, typename Eq = uint32_t, bool ZigZag = false>
pszerror psz_comp_l23r(
    T* const data,
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    dim3 const len3,
#elif defined(PSZ_USE_1API)
    sycl::range<3> const len3,
#endif
    PROPER_EB const eb, int const radius, Eq* const eq, void* _outlier,
    float* time_elapsed,
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    void* _stream
#elif defined(PSZ_USE_1API)
    void* _queue
#endif
);

#endif /* EBA07D6C_FD5C_446C_9372_F78DDB5E2B34 */
