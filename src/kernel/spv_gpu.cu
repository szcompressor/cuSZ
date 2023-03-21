/**
 * @file spv_gpu.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-29
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "../detail/spv_gpu.inl"
#include "kernel/spv_gpu.h"
#include "kernel/spv_gpu.hh"

#define SPV(Tliteral, Mliteral, T, M)                                                                              \
    void spv_gather_T##Tliteral##_M##Mliteral(                                                                     \
        T* in, size_t const in_len, T* d_val, uint32_t* d_idx, int* nnz, float* milliseconds, cudaStream_t stream) \
    {                                                                                                              \
        psz::detail::spv_gather<T, M>(in, in_len, d_val, d_idx, nnz, milliseconds, stream);                        \
    }                                                                                                              \
                                                                                                                   \
    void spv_scatter_T##Tliteral##_M##Mliteral(                                                                    \
        T* d_val, uint32_t* d_idx, int const nnz, T* decoded, float* milliseconds, cudaStream_t stream)            \
    {                                                                                                              \
        psz::detail::spv_scatter<T, M>(d_val, d_idx, nnz, decoded, milliseconds, stream);                          \
    }

SPV(ui8, ui32, uint8_t, uint32_t)
SPV(ui16, ui32, uint16_t, uint32_t)
SPV(ui32, ui32, uint32_t, uint32_t)
SPV(ui64, ui32, uint64_t, uint32_t)
SPV(fp32, ui32, float, uint32_t)
SPV(fp64, ui32, double, uint32_t)

#undef SPV

#define SPV(Tliteral, Mliteral, T, M)                                                                               \
    template <>                                                                                                     \
    void psz::spv_gather<T, M>(                                                                                     \
        T * in, size_t const in_len, T* d_val, uint32_t* d_idx, int* nnz, float* milliseconds, cudaStream_t stream) \
    {                                                                                                               \
        spv_gather_T##Tliteral##_M##Mliteral(in, in_len, d_val, d_idx, nnz, milliseconds, stream);                  \
    }                                                                                                               \
                                                                                                                    \
    template <>                                                                                                     \
    void psz::spv_scatter<T, M>(                                                                                    \
        T * d_val, uint32_t * d_idx, int const nnz, T* decoded, float* milliseconds, cudaStream_t stream)           \
    {                                                                                                               \
        spv_scatter_T##Tliteral##_M##Mliteral(d_val, d_idx, nnz, decoded, milliseconds, stream);                    \
    }

SPV(ui8, ui32, uint8_t, uint32_t)
SPV(ui16, ui32, uint16_t, uint32_t)
SPV(ui32, ui32, uint32_t, uint32_t)
SPV(ui64, ui32, uint64_t, uint32_t)
SPV(fp32, ui32, float, uint32_t)
SPV(fp64, ui32, double, uint32_t)

#undef SPV
