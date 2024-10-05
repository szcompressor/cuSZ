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

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)

namespace psz::cuhip {

template <typename T, typename E, psz_timing_mode TIMING, bool ZigZag = false>
pszerror GPU_c_lorenzo_nd_with_outlier(
    T* const data, dim3 const len3, PROPER_EB const eb, int const radius,
    E* const eq, void* _outlier, float* time_elapsed, void* _stream);

template <
    typename TIN, typename TOUT, bool ReverseProcess, psz_timing_mode TIMING>
pszerror GPU_lorenzo_prequant(
    TIN* const in, size_t const len, PROPER_EB const eb, TOUT* const out,
    float* time_elapsed, void* _stream);

}  // namespace psz::cuhip

#endif

#if defined(PSZ_USE_1API)

namespace psz::dpcpp {

template <typename T, typename E, psz_timing_mode TIMING, bool ZigZag = false>
pszerror GPU_c_lorenzo_nd_with_outlier(
    T* const data, sycl::range<3> const len3, PROPER_EB const eb,
    int const radius, E* const eq, void* _outlier, float* time_elapsed,
    void* _queue);

}  // namespace psz::dpcpp

#endif

#endif /* EBA07D6C_FD5C_446C_9372_F78DDB5E2B34 */
