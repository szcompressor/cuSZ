/**
 * @file hfbk_p1.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-03
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <cstdint>
#include "detail/hfbk_p1.cu.inl"
#include "hf/hf_bookg.hh"

// Codeword length
template __global__ void par_huffman::GPU_GenerateCL<uint32_t>(
    uint32_t*,
    uint32_t*,
    int,
    uint32_t*,
    int*,
    uint32_t*,
    int*,
    uint32_t*,
    int*,
    int*,
    uint32_t*,
    int*,
    int*,
    uint32_t*,
    int,
    int);

template __global__ void
par_huffman::GPU_GenerateCW<uint32_t, uint32_t>(uint32_t* CL, uint32_t* CW, uint32_t* first, uint32_t* entry, int size);
template __global__ void par_huffman::GPU_GenerateCW<uint32_t, unsigned long long>(
    uint32_t*           CL,
    unsigned long long* CW,
    unsigned long long* first,
    unsigned long long* entry,
    int                 size);