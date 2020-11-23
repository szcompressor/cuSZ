/**
 * @file histogram.cu
 * @author Cody Rivera (cjrivera1@crimson.ua.edu), Megan Hickman Fulp (mlhickm@g.clemson.edu)
 * @brief Fast histogramming from [Gómez-Luna et al. 2013]
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-02-16
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

// includes CUDA Runtime
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#include "ad_hoc_types.hh"
#include "histogram.cuh"

__global__ void data_process::reduce::NaiveHistogram(int input_data[], int output[], int N, int symbols_per_thread)
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

template <typename Input, typename Output_UInt>
__global__ void data_process::reduce::p2013Histogram(Input* input_data, Output_UInt* output, size_t N, int bins, int R)
{
    // TODO compile time type-checking
    extern __shared__ int Hs[/*(bins + 1) * R*/];

    const unsigned int warpid      = (int)(threadIdx.x / WARP_SIZE);
    const unsigned int lane        = threadIdx.x % WARP_SIZE;
    const unsigned int warps_block = blockDim.x / WARP_SIZE;

    const unsigned int off_rep = (bins + 1) * (threadIdx.x % R);

    const unsigned int begin = (N / warps_block) * warpid + WARP_SIZE * blockIdx.x + lane;
    unsigned int       end   = (N / warps_block) * (warpid + 1);
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

template __global__ void data_process::reduce::p2013Histogram<UI1, unsigned int>(UI1*, unsigned int*, size_t, int, int);
template __global__ void data_process::reduce::p2013Histogram<UI2, unsigned int>(UI2*, unsigned int*, size_t, int, int);
template __global__ void data_process::reduce::p2013Histogram<UI4, unsigned int>(UI4*, unsigned int*, size_t, int, int);
