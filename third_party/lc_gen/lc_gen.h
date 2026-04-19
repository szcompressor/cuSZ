#pragma once

void BITR_COMPRESS(uint8_t* input, size_t insize, uint8_t** output, size_t* outsize, float* time, void* stream);

void TCMS_COMPRESS(uint8_t* input, size_t insize, uint8_t** output, size_t* outsize, float* time, void* stream);

void RTR_COMPRESS(uint8_t* input, size_t insize, uint8_t** output, size_t* outsize, float* time, void* stream);

void BITR_DECOMPRESS(uint8_t* input, void** output, float* time);

void TCMS_DECOMPRESS(uint8_t* input, void** output, float* time);

void RTR_DECOMPRESS(uint8_t* input, void** output, float* time);
