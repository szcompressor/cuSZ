#ifndef PHF_HFBK_HH
#define PHF_HFBK_HH

#include <stdint.h>

#include "c_type.h"

template <typename T, typename H>
void phf_GPU_build_canonized_codebook(
    uint32_t* freq, int const bklen, H* book, uint8_t* revbook, int const revbook_nbyte,
    float* time, void* = nullptr);

template <typename E, typename H = uint32_t>
[[deprecated("use phf_CPU_build_canonized_codebook_v2")]] void phf_CPU_build_canonized_codebook_v1(
    uint32_t* freq, int const bklen, H* book, uint8_t* revbook, int const revbook_bytes,
    float* time);

template <typename E, typename H = uint32_t>
void phf_CPU_build_canonized_codebook_v2(
    uint32_t* freq, int const bklen, uint32_t* bk4, uint8_t* revbook, int const revbook_bytes,
    float* time);

#endif /* PHF_HFBK_HH */
