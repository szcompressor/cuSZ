// Cody Rivera <cjrivera1@crimson.ua.edu>
// Megan Hickman

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include "histogram.cuh"

__global__ void naiveHistogram(int input_data[], int output[], int N, int symbols_per_thread)
{
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

// const static unsigned int WARP_SIZE = 32;

#define MIN(a, b) ((a) < (b)) ? (a) : (b)

template <typename T, typename Q>
__global__ void p2013Histogram(T* input_data, Q* output, size_t N, int bins, int R)
{
    extern __shared__ int Hs[/*(bins + 1) * R*/];

    const unsigned int warpid      = (int)(threadIdx.x / WARP_SIZE);
    const unsigned int lane        = threadIdx.x % WARP_SIZE;
    const unsigned int warps_block = blockDim.x / WARP_SIZE;

    const unsigned int off_rep = (bins + 1) * (threadIdx.x % R);

    const unsigned int begin = (N / warps_block) * warpid + WARP_SIZE * blockIdx.x + lane;
    unsigned int end         = (N / warps_block) * (warpid + 1);
    const unsigned int step  = WARP_SIZE * gridDim.x;

    // final warp handles data outside of the warps_block partitions
    if (warpid >= warps_block - 1) end = N;

    for (unsigned int pos = threadIdx.x; pos < (bins + 1) * R; pos += blockDim.x) Hs[pos] = 0;

    __syncthreads();

    for (unsigned int i = begin; i < end; i += step) {
        int d = input_data[i];
        atomicAdd(&Hs[off_rep + d], 1);
    }

    __syncthreads();

    for (unsigned int pos = threadIdx.x; pos < bins; pos += blockDim.x) {
        int sum = 0;
        for (int base = 0; base < (bins + 1) * R; base += bins + 1) { sum += Hs[base + pos]; }
        atomicAdd(output + pos, sum);
    }
}

template __global__ void p2013Histogram<uint8_t, unsigned int>(uint8_t* input_data, unsigned int* output, size_t N, int bins, int R);
template __global__ void p2013Histogram<uint16_t, unsigned int>(uint16_t* input_data, unsigned int* output, size_t N, int bins, int R);
template __global__ void p2013Histogram<uint32_t, unsigned int>(uint32_t* input_data, unsigned int* output, size_t N, int bins, int R);
