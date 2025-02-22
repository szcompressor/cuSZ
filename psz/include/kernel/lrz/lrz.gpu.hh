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

#include <array>
#include <cstdint>

#include "cusz/type.h"

// TODO put f4/f8 clarification in the kernel impl. files
// #if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
// #define PROPER_EB f8
// #elif defined(PSZ_USE_1API)
// #define PROPER_EB f4
// #endif

using stdlen3 = std::array<size_t, 3>;
using Hf = u4;

namespace psz::module {

template <typename T, bool UseZigZag, typename Eq, bool EncodeInPlace = false>
int GPU_c_lorenzo_nd_with_outlier(
    T* const in_data, stdlen3 const data_len3, Eq* const out_eq, void* out_outlier, uint32_t* top1,
    f8 const ebx2, f8 const ebx2_r, uint16_t const radius, void* stream, Hf* pbk = nullptr,
    u1* pbk_tree_IDs = nullptr, Hf* pbk_bitstream = nullptr, u2* pbk_bit = nullptr,
    u4* pbk_entries = nullptr, size_t* pbk_loc = nullptr,  //
    Eq* const brval = nullptr, u4* const bridx = nullptr, u4* const brnum = nullptr);

template <typename T, bool UseZigZag, typename Eq>
int GPU_x_lorenzo_nd(
    Eq* const in_eq, T* const in_outlier, T* const out_data, stdlen3 const data_len3,
    f8 const ebx2, f8 const ebx2_r, uint16_t const radius, void* stream);

template <typename TIN, typename TOUT, bool ReverseProcess>
int GPU_lorenzo_prequant(
    TIN* const in, size_t const len, f8 const ebx2, f8 const ebx2_r, TOUT* const out,
    void* _stream);

}  // namespace psz::module

#endif /* PSZ_MODULE_LRZ_GPU_HH */