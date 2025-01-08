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

#ifndef PSZ_MODULE_LRZ_GPU_HH
#define PSZ_MODULE_LRZ_GPU_HH

#include <cstdint>

#include "cusz/suint.hh"
#include "cusz/type.h"
#include "mem/cxx_sp_gpu.h"
#include "port.hh"

namespace psz::module {

template <typename T, bool UseZigZag, typename Eq>
pszerror GPU_c_lorenzo_nd_with_outlier(
    T* const in_data, std::array<size_t, 3> const data_len3, Eq* const out_eq, void* out_outlier,
    f8 const eb, uint16_t const radius, f4* time_elapsed, void* stream);

template <typename T, bool UseZigZag, typename Eq>
pszerror GPU_x_lorenzo_nd(
    Eq* const in_eq, T* const in_outlier, T* const out_data, std::array<size_t, 3> const data_len3,
    f8 const eb, uint16_t const radius, f4* time_elapsed, void* stream);

template <typename TIN, typename TOUT, bool ReverseProcess>
pszerror GPU_lorenzo_prequant(
    TIN* const in, size_t const len, PROPER_EB const eb, TOUT* const out, float* time_elapsed,
    void* _stream);

}  // namespace psz::module

#endif /* PSZ_MODULE_LRZ_GPU_HH */