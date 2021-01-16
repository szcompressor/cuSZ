/**
 * @file cusz_dualquant.cuh
 * @author Jiannan Tian
 * @brief Dual-Quantization method of cuSZ (header).
 * @version 0.2
 * @date 2021-01-16
 * (create) 19-09-23; (release) 2020-09-20; (rev1) 2021-01-16
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_DUALQUANT_CUH
#define CUSZ_DUALQUANT_CUH

#include <cuda_runtime.h>
#include <cstddef>

#include "metadata.hh"

extern __shared__ char scratch[];
// extern __shared__ float s2df[][16 + 1];  // TODO double type
// extern __shared__ float s3df[][8+ 1][8+ 1];

namespace cusz {
namespace predictor_quantizer {

// clang-format off
// version 1
template <typename Data, typename Quant> __global__ void c_lorenzo_1d1l(Data*, Quant*, size_t const*, double const*);
template <typename Data, typename Quant> __global__ void c_lorenzo_2d1l(Data*, Quant*, size_t const*, double const*);
template <typename Data, typename Quant> __global__ void c_lorenzo_3d1l(Data*, Quant*, size_t const*, double const*);

template <typename Data, typename Quant> __global__ void x_lorenzo_1d1l(Data*, Data*, Quant*, size_t const*, double);
template <typename Data, typename Quant> __global__ void x_lorenzo_2d1l(Data*, Data*, Quant*, size_t const*, double);
template <typename Data, typename Quant> __global__ void x_lorenzo_3d1l(Data*, Data*, Quant*, size_t const*, double);

// version 2
template <typename Data, typename Quant> __global__ void c_lorenzo_1d1l_v2(lorenzo_zip, Data*, Quant*);
template <typename Data, typename Quant> __global__ void c_lorenzo_2d1l_v2(lorenzo_zip, Data*, Quant*);
template <typename Data, typename Quant> __global__ void c_lorenzo_3d1l_v2(lorenzo_zip, Data*, Quant*);

template <typename Data, typename Quant> __global__ void x_lorenzo_1d1l_v2(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_2d1l_v2(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_3d1l_v2(lorenzo_unzip, Data*, Data*, Quant*);

// version 3
// 1D and unzip remain the same
template <typename Data, typename Quant> __global__ void c_lorenzo_2d1l_v3(lorenzo_zip, Data*, Quant*);
template <typename Data, typename Quant> __global__ void c_lorenzo_3d1l_v3(lorenzo_zip, Data*, Quant*);

// clang-format on

}  // namespace predictor_quantizer
}  // namespace cusz

#endif
