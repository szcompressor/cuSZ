/**
 * @file spv_gpu.h
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-29
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef B1B21251_C3C3_4BC1_B4E0_75D9D86EE7F3
#define B1B21251_C3C3_4BC1_B4E0_75D9D86EE7F3

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime.h>
#include <stdint.h>

#define SPV(Tliteral, Mliteral, T, M)                                                                               \
    void spv_gather_T##Tliteral##_M##Mliteral(                                                                      \
        T* in, size_t const in_len, T* d_val, uint32_t* d_idx, int* nnz, float* milliseconds, cudaStream_t stream); \
                                                                                                                    \
    void spv_scatter_T##Tliteral##_M##Mliteral(                                                                     \
        T* d_val, uint32_t* d_idx, int const nnz, T* decoded, float* milliseconds, cudaStream_t stream);

SPV(ui8, ui32, uint8_t, uint32_t)
SPV(ui16, ui32, uint16_t, uint32_t)
SPV(ui32, ui32, uint32_t, uint32_t)
SPV(ui64, ui32, uint64_t, uint32_t)
SPV(fp32, ui32, float, uint32_t)
SPV(fp64, ui32, double, uint32_t)

#undef SPV

#ifdef __cplusplus
}
#endif

#endif /* B1B21251_C3C3_4BC1_B4E0_75D9D86EE7F3 */
