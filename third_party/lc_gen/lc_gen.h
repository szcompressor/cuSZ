#pragma once

#include <cstddef>
#include <cstdint>

#include "lc_buf.h"
#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "utils/err.hh"
#endif

namespace LC_Connector {

#if defined(__CUDACC__)
struct Config {
  int SMs;
  int mTpSM;
  int blocks;
};

static inline Config config(const int TPB)
{
  CHECK_GPU(cudaSetDevice(0));
  cudaDeviceProp deviceProp;
  CHECK_GPU(cudaGetDeviceProperties(&deviceProp, 0));

  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;

  return {SMs, mTpSM, SMs * (mTpSM / TPB)};
}
#endif

void BITR_COMPRESS(uint8_t* input, size_t insize, psz::LC_Buf* buf, size_t* outsize, void* stream);
void TCMS_COMPRESS(uint8_t* input, size_t insize, psz::LC_Buf* buf, size_t* outsize, void* stream);
void RTR_COMPRESS(uint8_t* input, size_t insize, psz::LC_Buf* buf, size_t* outsize, void* stream);

void BITR_DECOMPRESS(uint8_t* input, psz::LC_Buf* buf, void* stream);
void TCMS_DECOMPRESS(uint8_t* input, psz::LC_Buf* buf, void* stream);
void RTR_DECOMPRESS(uint8_t* input, psz::LC_Buf* buf, void* stream);

}  // namespace LC_Connector

namespace lc_c = LC_Connector;
