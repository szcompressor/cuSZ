/**
 * @file spv_gpu.hh
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

namespace accsz {

template <typename T, typename M>
void spv_gather(
    T*           in,
    size_t const in_len,
    T*           d_val,
    uint32_t*    d_idx,
    int*         nnz,
    float*       milliseconds,
    cudaStream_t stream);

template <typename T, typename M>
void spv_scatter(T* d_val, uint32_t* d_idx, int const nnz, T* decoded, float* milliseconds, cudaStream_t stream);

}  // namespace accsz

#endif /* A54D2009_1D4F_4113_9E26_9695A3669224 */
