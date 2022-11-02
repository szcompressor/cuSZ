/**
 * @file hf_bookg.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-03
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "detail/hf_bookg.inl"
#include "hf/hf_bookg.hh"

/********************************************************************************/
// instantiate wrapper

#define PAR_BOOK(T, H) \
    template void asz::parallel_get_codebook<T, H>(uint32_t*, int const, H*, uint8_t*, int const, float*, cudaStream_t);

PAR_BOOK(uint8_t, uint32_t);
PAR_BOOK(uint16_t, uint32_t);
PAR_BOOK(uint32_t, uint32_t);
PAR_BOOK(float, uint32_t);

PAR_BOOK(uint8_t, uint64_t);
PAR_BOOK(uint16_t, uint64_t);
PAR_BOOK(uint32_t, uint64_t);
PAR_BOOK(float, uint64_t);

PAR_BOOK(uint8_t, unsigned long long);
PAR_BOOK(uint16_t, unsigned long long);
PAR_BOOK(uint32_t, unsigned long long);
PAR_BOOK(float, unsigned long long);

// #define C_GPUPAR_CODEBOOK(Tliteral, Hliteral, Mliteral, T, H, M)                                                 \
//     cusz_error_status claunch_gpu_parallel_build_codebook_T##Tliteral##_H##Hliteral##_M##Mliteral(               \
//         uint32_t* freq, H* book, int const booklen, uint8_t* revbook, int const revbook_nbyte, float* time_book, \
//         cudaStream_t stream);

// C_GPUPAR_CODEBOOK(ui8, ui32, ui32, uint8_t, uint32_t, uint32_t);
// C_GPUPAR_CODEBOOK(ui16, ui32, ui32, uint8_t, uint32_t, uint32_t);
// C_GPUPAR_CODEBOOK(ui32, ui32, ui32, uint8_t, uint32_t, uint32_t);
// C_GPUPAR_CODEBOOK(ui8, ui64, ui32, uint8_t, uint64_t, uint32_t);
// C_GPUPAR_CODEBOOK(ui16, ui64, ui32, uint8_t, uint64_t, uint32_t);
// C_GPUPAR_CODEBOOK(ui32, ui64, ui32, uint8_t, uint64_t, uint32_t);
// C_GPUPAR_CODEBOOK(ui8, ul, ui32, uint8_t, unsigned long, uint32_t);
// C_GPUPAR_CODEBOOK(ui16, ul, ui32, uint8_t, unsigned long, uint32_t);
// C_GPUPAR_CODEBOOK(ui32, ul, ui32, uint8_t, unsigned long, uint32_t);
//
// #undef C_GPUPAR_CODEBOOK

#undef PAR_BOOK
