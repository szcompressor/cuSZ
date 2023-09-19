/**
 * @file hfbk_p2.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-03
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "detail/hfbk_p2.cu.inl"
#include "hf/hf_bk.hh"
#include "hf/hf_bookg.hh"

#define HF_BOOK_CU(T, H)                                                     \
  template <>                                                                \
  void psz::hf_buildbook<CUDA, T, H>(                                        \
      uint32_t * freq, int const bklen, H* book, uint8_t* revbook,           \
      int const revbook_bytes, float* time, void* stream)                       \
  {                                                                          \
    psz::hf_buildbook_cu<T, H>(                                              \
        freq, bklen, book, revbook, revbook_bytes, time, (cudaStream_t)stream); \
  }
// 23-06-04 restricted to u4 for quantization code

// PAR_BOOK(uint8_t, uint32_t);
// PAR_BOOK(uint16_t, uint32_t);
HF_BOOK_CU(uint32_t, uint32_t);

// PAR_BOOK(uint8_t, unsigned long long);
// PAR_BOOK(uint16_t, unsigned long long);
HF_BOOK_CU(uint32_t, unsigned long long);

#undef HF_BOOK_CU
