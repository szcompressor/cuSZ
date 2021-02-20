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

// clang-format off
namespace cusz { namespace predictor_quantizer {
template <typename Data, typename Quant> __global__ void c_lorenzo_1d1l(lorenzo_zip, Data*, Quant*);
template <typename Data, typename Quant> __global__ void c_lorenzo_2d1l(lorenzo_zip, Data*, Quant*);
template <typename Data, typename Quant> __global__ void c_lorenzo_3d1l(lorenzo_zip, Data*, Quant*);

namespace prototype { // easy for algorithmic description
template <typename Data, typename Quant> __global__ void x_lorenzo_1d1l(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_2d1l(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_3d1l(lorenzo_unzip, Data*, Data*, Quant*); }

namespace legacy {
template <typename Data, typename Quant> __global__ void x_lorenzo_2d1l_16x16_v0(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_3d1l_8x8x8_v0(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_3d1l_8x8x8_v1(lorenzo_unzip, Data*, Data*, Quant*); }

template <typename Data, typename Quant> __global__ void x_lorenzo_1d1l_cub(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_2d1l_16x16_v1(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_3d1l_8x8x8_v2(lorenzo_unzip, Data*, Data*, Quant*);
}}
// clang-format on
#endif
