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

#include <cuda_runtime.h>
#include <cstdio>

#include "hist.cuh"
#include "type_aliasing.hh"

#define MIN(a, b) ((a) < (b)) ? (a) : (b)

#define tix threadIdx.x
#define tiy threadIdx.y
#define tiz threadIdx.z
#define bix blockIdx.x
#define biy blockIdx.y
#define biz blockIdx.z
#define bdx blockDim.x
#define bdy blockDim.y
#define bdz blockDim.z

namespace kernel = data_process::reduce;

__global__ void kernel::NaiveHistogram(int input_data[], int output[], int N, int symbols_per_thread)
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

template <typename Input_UInt, typename Output_UInt>
__global__ void kernel::p2013Histogram(Input_UInt* input_data, Output_UInt* output, size_t N, int bins, int R)
{
    static_assert(
        std::is_same<Input_UInt, UI1>::value         //
            or std::is_same<Input_UInt, UI2>::value  //
            or std::is_same<Input_UInt, UI4>::value,
        "Input_UInt Must be Unsigned Integer type of {1,2,4} bytes");

    extern __shared__ int Hs[/*(bins + 1) * R*/];

    const unsigned int warp_id     = (int)(tix / WARP_SIZE);
    const unsigned int lane        = tix % WARP_SIZE;
    const unsigned int warps_block = bdx / WARP_SIZE;
    const unsigned int off_rep     = (bins + 1) * (tix % R);
    const unsigned int begin       = (N / warps_block) * warp_id + WARP_SIZE * blockIdx.x + lane;
    unsigned int       end         = (N / warps_block) * (warp_id + 1);
    const unsigned int step        = WARP_SIZE * gridDim.x;

    // final warp handles data outside of the warps_block partitions
    if (warp_id >= warps_block - 1) end = N;

    for (unsigned int pos = tix; pos < (bins + 1) * R; pos += bdx) Hs[pos] = 0;
    __syncthreads();

    for (unsigned int i = begin; i < end; i += step) {
        int d = input_data[i];
        atomicAdd(&Hs[off_rep + d], 1);
    }
    __syncthreads();

    for (unsigned int pos = tix; pos < bins; pos += bdx) {
        int sum = 0;
        for (int base = 0; base < (bins + 1) * R; base += bins + 1) { sum += Hs[base + pos]; }
        atomicAdd(output + pos, sum);
    }
}

// TODO necessary for 4-byte?
template __global__ void kernel::p2013Histogram<UI1, unsigned int>(UI1*, unsigned int*, size_t, int, int);
template __global__ void kernel::p2013Histogram<UI2, unsigned int>(UI2*, unsigned int*, size_t, int, int);
template __global__ void kernel::p2013Histogram<UI4, unsigned int>(UI4*, unsigned int*, size_t, int, int);

template <typename Input_Int, typename Output_UInt>
__global__ void
kernel::p2013Histogram_int_input(Input_Int* input_data, Output_UInt* output, size_t N, int bins, int R, int radius)
{
    static_assert(
        std::is_same<Input_Int, I1>::value         //
            or std::is_same<Input_Int, I2>::value  //
            or std::is_same<Input_Int, I4>::value,
        "Input_Int Must be Signed Integer type of {1,2,4} bytes");

    extern __shared__ int Hs[/*(bins + 1) * R*/];

    const unsigned int warp_id     = (int)(tix / WARP_SIZE);
    const unsigned int lane        = tix % WARP_SIZE;
    const unsigned int warps_block = bdx / WARP_SIZE;
    const unsigned int off_rep     = (bins + 1) * (tix % R);
    const unsigned int begin       = (N / warps_block) * warp_id + WARP_SIZE * blockIdx.x + lane;
    unsigned int       end         = (N / warps_block) * (warp_id + 1);
    const unsigned int step        = WARP_SIZE * gridDim.x;

    // final warp handles data outside of the warps_block partitions
    if (warp_id >= warps_block - 1) end = N;

    for (unsigned int pos = tix; pos < (bins + 1) * R; pos += bdx) Hs[pos] = 0;
    __syncthreads();

    for (unsigned int i = begin; i < end; i += step) {
        int d = input_data[i] + radius;
        atomicAdd(&Hs[off_rep + d], 1);
    }
    __syncthreads();

    for (unsigned int pos = tix; pos < bins; pos += bdx) {
        int sum = 0;
        for (int base = 0; base < (bins + 1) * R; base += bins + 1) { sum += Hs[base + pos]; }
        atomicAdd(output + pos, sum);
    }
}

template __global__ void kernel::p2013Histogram_int_input<I1, unsigned int>(I1*, unsigned int*, size_t, int, int, int);
template __global__ void kernel::p2013Histogram_int_input<I2, unsigned int>(I2*, unsigned int*, size_t, int, int, int);
template __global__ void kernel::p2013Histogram_int_input<I4, unsigned int>(I4*, unsigned int*, size_t, int, int, int);
