#ifndef MEMOPS_HH
#define MEMOPS_HH

/**
 * @file future_memops.hh
 * @author Jiannan Tian
 * @brief Internal use: wavefront padding.
 * @version 0.1
 * @date 2020-09-20
 * Created on 2019-09-20
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cstdint>

namespace memops {

/*
 *          s1
 *    _ |---------|       v
 *    |  o o o o o _ _ _  |
 * s0 |  o o o o o _ _ _  |
 *    |  o o o o o _ _ _  | gs0
 *    -  _ _ _ _ _ _ _ _  |
 *       _ _ _ _ _ _ _ _  |
 *      >---------------< ^
 *             gs1
 *    In 2D case, gs1 is ``leading dimension'' to support subblocks
 *
 */
template <typename DATA_T, size_t PADDING>
inline void pad2d(DATA_T* src, DATA_T* dst, size_t s0, size_t s1, size_t gs0, size_t gs1)
{
    for (size_t i0 = 0; i0 < s0; i0++) {
        DATA_T* _src = src + i0 * gs1;
        DATA_T* _dst = dst + (i0 * (gs1 + PADDING) + PADDING);
        std::copy(_src, _src + s1, _dst);
    }
}

template <typename DATA_T, size_t PADDING>
inline void depad2d(DATA_T* src, DATA_T* dst, size_t s0, size_t s1, size_t gs0, size_t gs1)
{
    for (size_t i0 = 0; i0 < s0; i0++) {
        DATA_T* _src = src + i0 * gs1;
        DATA_T* _dst = dst + (i0 * (gs1 - PADDING) - PADDING);
        std::copy(_src, _src + s1, _dst);
    }
}

void transpose_omp(float* src, float* dst, const int N, const int M)
{
    //#pragma omp parallel for
    for (int n = 0; n < N * M; n++) {
        int i  = n / N;
        int j  = n % N;
        dst[n] = src[M * j + i];
    }
}

}  // namespace memops

#endif  // MEMOPS_HH
