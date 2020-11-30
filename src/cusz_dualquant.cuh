/**
 * @file cusz_dualquant.cuh
 * @author Jiannan Tian
 * @brief Dual-Quantization method of cuSZ (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on 19-09-23
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_DUALQUANT_CUH
#define CUSZ_DUALQUANT_CUH

#include <cuda_runtime.h>
#include <cstddef>

extern __shared__ char scratch[];
// extern __shared__ float s2df[][16 + 1];  // TODO double type
// extern __shared__ float s3df[][8+ 1][8+ 1];

namespace cusz {
namespace PdQ {

template <typename T, typename Q, int B = 32>
__global__ void c_lorenzo_1d1l(T* data, Q* code, size_t const* dims, double const* precisions);

template <typename T, typename Q, int B = 16>
__global__ void c_lorenzo_2d1l(T* data, Q* code, size_t const* dims, double const* precisions);

template <typename T, typename Q, int B = 8>
__global__ void c_lorenzo_3d1l(T* data, Q* code, size_t const* dims, double const* precisions);

// use const memory
template <typename T, typename Q, int B = 32>
__global__ void c_lorenzo_1d1l_cmem(T* data, Q* code);

template <typename T, typename Q, int B = 16>
__global__ void c_lorenzo_2d1l_cmem(T* data, Q* code);

template <typename T, typename Q, int B = 8>
__global__ void c_lorenzo_3d1l_cmem(T* data, Q* code);

////////////////////////////////////////////////////////////////////////////////////////////////////
//   ^                 decompression |
//   |compression                    v
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename Q, int B = 32>
__global__ void x_lorenzo_1d1l(T* xdata, T* outlier, Q* bcode, size_t const* dims, double val_2eb);

template <typename T, typename Q, int B = 16>
__global__ void x_lorenzo_2d1l(T* xdata, T* outlier, Q* bcode, size_t const* dims, double val_2eb);

template <typename T, typename Q, int B = 8>
__global__ void x_lorenzo_3d1l(T* xdata, T* outlier, Q* bcode, size_t const* dims, double val_2eb);

}  // namespace PdQ
}  // namespace cusz

#endif
