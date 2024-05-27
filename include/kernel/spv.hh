/**
 * @file spv_cu.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-29
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef A54D2009_1D4F_4113_9E26_9695A3669224
#define A54D2009_1D4F_4113_9E26_9695A3669224

#include <cstdint>

#include "cusz/type.h"

namespace psz {

template <pszpolicy P, typename T, typename M = u4>
void spv_gather(
    T* in, size_t const in_len, T* d_val, M* d_idx, int* nnz, f4* milliseconds,
    void* stream);

template <pszpolicy P, typename T, typename M = u4>
void spv_scatter(
    T* d_val, M* d_idx, int const nnz, T* decoded, f4* milliseconds,
    void* stream);

template <pszpolicy P, typename T, typename Criterion, typename M = u4>
void spv_gather_naive(
    T* in, size_t const in_len, int const radius, T* cval, M* cidx, int* cn,
    Criterion c, f4* milliseconds, void* stream);

template <pszpolicy P, typename T, typename M = u4>
void spv_scatter_naive(
    T* d_val, M* d_idx, int const nnz, T* decoded, f4* milliseconds,
    void* stream);

}  // namespace psz

#endif /* A54D2009_1D4F_4113_9E26_9695A3669224 */
