/**
 * @file l23.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-01
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef B297267F_4731_48DB_8128_BBD202027EB7
#define B297267F_4731_48DB_8128_BBD202027EB7

#include <cstdint>

#include "cusz/suint.hh"
#include "cusz/type.h"
#include "mem/compact.hh"
#include "port.hh"

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)

namespace psz::cuhip {

template <typename T, typename Eq, bool ZigZag = false>
pszerror GPU_c_lorenzo_nd_with_outlier(
    T* const in_data, dim3 const data_len3, Eq* const out_eq,
    void* out_outlier, PROPER_EB const eb, uint16_t const radius,
    f4* time_elapsed, void* stream);

template <typename T, typename Eq>
pszerror GPU_x_lorenzo_nd(
    Eq* const in_eq, T* const in_outlier, T* const out_data,
    dim3 const data_len3, PROPER_EB const eb, uint16_t const radius,
    f4* time_elapsed, void* stream);

template <typename TIN, typename TOUT, bool ReverseProcess>
pszerror GPU_lorenzo_prequant(
    TIN* const in, size_t const len, PROPER_EB const eb, TOUT* const out,
    float* time_elapsed, void* _stream);

}  // namespace psz::cuhip

#endif

#if defined(PSZ_USE_1API)

namespace psz::dpcpp {

template <typename T, typename Eq, psz_timing_mode TIMING, bool ZigZag = false>
pszerror GPU_c_lorenzo_nd_with_outlier(
    T* const data, sycl::range<3> const len3, PROPER_EB const eb,
    int const radius, Eq* const eq, void* _outlier, float* time_elapsed,
    void* _queue);

template <typename T, typename E, psz_timing_mode TIMING = SYNC_BY_STREAM>
pszerror GPU_x_lorenzo_nd(
    E* eq, sycl::range<3> const len3, T* outlier, PROPER_EB const eb,
    int const radius, T* xdata, float* time_elapsed, void* stream);

}  // namespace psz::dpcpp

#endif

#endif /* B297267F_4731_48DB_8128_BBD202027EB7 */
