// This file compiles a specialzed pipeline based on LC framework.
// See copyright and license (BSD 3-Clause) in `third_party/lc`.

#ifndef NDEBUG
#define NDEBUG
#endif

using byte = unsigned char;
static const int CS =
    1024 * 16;  // chunk size (in bytes) [must be multiple of 8]
static const int TPB =
    512;  // threads per block [must be power of 2 and at least 128]
#if defined(__AMDGCN_WAVEFRONT_SIZE) && (__AMDGCN_WAVEFRONT_SIZE == 64)
#define WS 64
#else
#define WS 32
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

#include "../lc/include/max_scan.h"
#include "../lc/include/prefix_sum.h"
#include "../lc/include/sum_reduction.h"
//
#include "../lc/components/d_BIT_1.h"
#include "../lc/components/d_RRE_1.h"
#include "../lc/components/d_TCMS_1.h"
#include "lc_gen/lc_gen.h"

// copy (len) bytes from global memory (source) to shared memory (destination)
// using separate shared memory buffer (temp) destination and temp must we word
// aligned, accesses up to CS + 3 bytes in temp
static inline __device__ void g2s(
    void* const __restrict__ destination,
    const void* const __restrict__ source, const int len,
    void* const __restrict__ temp)
{
  const int tid = threadIdx.x;
  const byte* const __restrict__ input = (byte*)source;
  if (len < 128) {
    byte* const __restrict__ output = (byte*)destination;
    if (tid < len) output[tid] = input[tid];
  }
  else {
    const int nonaligned = (int)(size_t)input;
    const int wordaligned = (nonaligned + 3) & ~3;
    const int linealigned = (nonaligned + 127) & ~127;
    const int bcnt = wordaligned - nonaligned;
    const int wcnt = (linealigned - wordaligned) / 4;
    int* const __restrict__ out_w = (int*)destination;
    if (bcnt == 0) {
      const int* const __restrict__ in_w = (int*)input;
      byte* const __restrict__ out = (byte*)destination;
      if (tid < wcnt) out_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < len / 4; i += TPB) { out_w[i] = in_w[i]; }
      if (tid < (len & 3)) {
        const int i = len - 1 - tid;
        out[i] = input[i];
      }
    }
    else {
      const int offs = 4 - bcnt;  //(4 - bcnt) & 3;
      const int shift = offs * 8;
      const int rlen = len - bcnt;
      const int* const __restrict__ in_w = (int*)&input[bcnt];
      byte* const __restrict__ buffer = (byte*)temp;
      byte* const __restrict__ buf = (byte*)&buffer[offs];
      int* __restrict__ buf_w =
          (int*)&buffer[4];  //(int*)&buffer[(bcnt + 3) & 4];
      if (tid < bcnt) buf[tid] = input[tid];
      if (tid < wcnt) buf_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < rlen / 4; i += TPB) { buf_w[i] = in_w[i]; }
      if (tid < (rlen & 3)) {
        const int i = len - 1 - tid;
        buf[i] = input[i];
      }
      __syncthreads();
      buf_w = (int*)buffer;
      for (int i = tid; i < (len + 3) / 4; i += TPB) {
        out_w[i] = __funnelshift_r(buf_w[i], buf_w[i + 1], shift);
      }
    }
  }
}

static __device__ int g_chunk_counter;

static __global__ void d_reset() { g_chunk_counter = 0; }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
    void d_decode_tcms(
        const byte* const __restrict__ input, byte* const __restrict__ output,
        int* const __restrict__ g_outsize)
{
  // allocate shared memory buffer
  __shared__ long long chunk[3 * (CS / sizeof(long long))];
  const int last = 3 * (CS / sizeof(long long)) - 2 - WS;

  // input header
  int* const head_in = (int*)input;
  const int outsize = head_in[0];

  // initialize
  const int chunks = (outsize + CS - 1) / CS;  // round up
  unsigned short* const size_in = (unsigned short*)&head_in[1];
  byte* const data_in = (byte*)&size_in[chunks];

  // loop over chunks
  const int tid = threadIdx.x;
  int prevChunkID = 0;
  int prevOffset = 0;
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(&g_chunk_counter, 1);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const int chunkID = chunk[last];
    const int base = chunkID * CS;
    if (base >= outsize) break;

    // compute sum of all prior csizes (start where left off in previous
    // iteration)
    int sum = 0;
    for (int i = prevChunkID + tid; i < chunkID; i += TPB) {
      sum += (int)size_in[i];
    }
    int csize = (int)size_in[chunkID];
    const int offs =
        prevOffset + block_sum_reduction(sum, (int*)&chunk[last + 1]);
    prevChunkID = chunkID;
    prevOffset = offs;

    // create the 3 shared memory buffers
    byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
    byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
    byte* temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

    // load chunk
    g2s(in, &data_in[offs], csize, out);
    byte* tmp = in;
    in = out;
    out = tmp;
    __syncthreads();  // chunk produced, chunk[last] consumed

    // decode
    const int osize = min(CS, outsize - base);
    if (csize < osize) {
      byte* tmp;
      tmp = in;
      in = out;
      out = tmp;
      d_iRRE_1(csize, in, out, temp);
      __syncthreads();
      tmp = in;
      in = out;
      out = tmp;
      d_iBIT_1(csize, in, out, temp);
      __syncthreads();
      tmp = in;
      in = out;
      out = tmp;
      d_iTCMS_1(csize, in, out, temp);
      __syncthreads();
    }

    if (csize != osize) {
      printf(
          "ERROR: csize %d doesn't match osize %d in chunk %d\n\n", csize,
          osize, chunkID);
      __trap();
    }
    long long* const output_l = (long long*)&output[base];
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += TPB) { output_l[i] = out_l[i]; }
    const int extra = osize % 8;
    if (tid < extra)
      output[base + osize - extra + tid] = out[osize - extra + tid];
  } while (true);

  if ((blockIdx.x == 0) && (tid == 0)) { *g_outsize = outsize; }
}

struct GPUTimer {
  cudaEvent_t beg, end;
  GPUTimer()
  {
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
  }
  ~GPUTimer()
  {
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
  }
  void start() { cudaEventRecord(beg, 0); }
  double stop()
  {
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float ms;
    cudaEventElapsedTime(&ms, beg, end);
    return ms;
  }
};

static void CheckCuda(const int line)
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(
        stderr, "CUDA error %d on line %d: %s\n\n", e, line,
        cudaGetErrorString(e));
    throw std::runtime_error("LC error");
  }
}

void TCMS_DECOMPRESS(uint8_t* input, void** output, float* time)
{
  int pre_size;
  cudaMemcpy(&pre_size, input, sizeof(int), cudaMemcpyDeviceToHost);

  // get GPU info
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
    fprintf(stderr, "ERROR: no CUDA capable device detected\n\n");
    throw std::runtime_error("LC error");
  }
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int blocks = SMs * (mTpSM / TPB);
  CheckCuda(__LINE__);

  // allocate GPU memory
  byte* d_decoded;
  cudaMalloc((void**)&d_decoded, pre_size);
  int* d_decsize;
  cudaMalloc((void**)&d_decsize, sizeof(int));
  CheckCuda(__LINE__);

  // warm up
  byte* d_decoded_dummy;
  cudaMalloc((void**)&d_decoded_dummy, pre_size);
  int* d_decsize_dummy;
  cudaMalloc((void**)&d_decsize_dummy, sizeof(int));
  d_decode_tcms<<<blocks, TPB>>>(input, d_decoded_dummy, d_decsize_dummy);
  cudaFree(d_decoded_dummy);
  cudaFree(d_decsize_dummy);

  // time GPU decoding
  GPUTimer dtimer;
  dtimer.start();
  d_reset<<<1, 1>>>();
  d_decode_tcms<<<blocks, TPB>>>(input, d_decoded, d_decsize);
  *time = (float)dtimer.stop();
  CheckCuda(__LINE__);
  *output = d_decoded;
}
