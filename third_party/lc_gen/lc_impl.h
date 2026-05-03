#pragma once

#include <cstddef>
#include <cstdint>

#include "lc_buf.h"

#if defined(__CUDACC__)
__global__ void d_reset_tcms_comp();
__global__ void d_reset_rtr_comp();
__global__ void d_reset_bitr_comp();
__global__ void d_reset_tcms_decomp();
__global__ void d_reset_rtr_decomp();
__global__ void d_reset_bitr_decomp();

__global__ void d_encode_tcms(
    const unsigned char* input, int insize, unsigned char* output, int* outsize, int* fullcarry);
__global__ void d_encode_rtr(
    const unsigned char* input, int insize, unsigned char* output, int* outsize, int* fullcarry);
__global__ void d_encode_bitr(
    const unsigned char* input, int insize, unsigned char* output, int* outsize, int* fullcarry);

__global__ void d_decode_tcms(const unsigned char* input, unsigned char* output, int* g_outsize);
__global__ void d_decode_rtr(const unsigned char* input, unsigned char* output, int* g_outsize);
__global__ void d_decode_bitr(const unsigned char* input, unsigned char* output, int* g_outsize);
#endif
