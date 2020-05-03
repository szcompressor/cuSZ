// by Cody Rivera <cjrivera1@crimson.ua.edu>

#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cstdio>

// Useful macros
#define cudaErrchk(ans) \
    { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDAassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/* Cuda kernel declarations */

__global__ void naiveHistogram(int input_data[], int output[], int N, int symbols_per_thread) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int j;
    if (i * symbols_per_thread < N) {  // if there is a symbol to count,
        for (j = i * symbols_per_thread; j < (i + 1) * symbols_per_thread; j++) {
            if (j < N) {
                unsigned int item = input_data[j];  // Symbol to count
                atomicAdd(&output[item], 1);        // update bin count by 1
            }
        }
    }
}

// Optimized 2007

//#ifndef WARP_SIZE
//#define WARP_SIZE 32
//#endif
const static unsigned int WARP_SIZE = 32;

#define MIN(a, b) ((a) < (b)) ? (a) : (b)

// Optimized 2013
/* Copied from J. Gomez-Luna et al */
// TODO int only??
template <typename T, typename Q>
__global__ void p2013Histogram(T* input_data, Q* output, size_t N, int bins, int R) {
    extern __shared__ int Hs[/*(bins + 1) * R*/];

    const unsigned int warpid      = (int)(threadIdx.x / WARP_SIZE);
    const unsigned int lane        = threadIdx.x % WARP_SIZE;
    const unsigned int warps_block = blockDim.x / WARP_SIZE;

    const unsigned int off_rep = (bins + 1) * (threadIdx.x % R);

    const unsigned int begin = (N / warps_block) * warpid + WARP_SIZE * blockIdx.x + lane;
    const unsigned int end   = (N / warps_block) * (warpid + 1);
    const unsigned int step  = WARP_SIZE * gridDim.x;

    for (unsigned int pos = threadIdx.x; pos < (bins + 1) * R; pos += blockDim.x) Hs[pos] = 0;

    __syncthreads();

    for (unsigned int i = begin; i < end; i += step) {
        int d = input_data[i];
        atomicAdd(&Hs[off_rep + d], 1);
    }

    __syncthreads();

    for (unsigned int pos = threadIdx.x; pos < bins; pos += blockDim.x) {
        int sum = 0;
        for (int base = 0; base < (bins + 1) * R; base += bins + 1) {
            sum += Hs[base + pos];
        }
        atomicAdd(output + pos, sum);
    }
}

#endif
