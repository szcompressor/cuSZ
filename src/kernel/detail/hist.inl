/**
 * @file hist.inl
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

#ifndef D69BE972_2A8C_472E_930F_FFAB041F3F2B
#define D69BE972_2A8C_472E_930F_FFAB041F3F2B

#include <cuda_runtime.h>
#include <cstdio>
#include <limits>

#include "type_traits.hh"
#include "utils/config.hh"
#include "utils/timer.h"

#define MIN(a, b) ((a) < (b)) ? (a) : (b)
const static unsigned int WARP_SIZE = 32;

#define tix threadIdx.x
#define tiy threadIdx.y
#define tiz threadIdx.z
#define bix blockIdx.x
#define biy blockIdx.y
#define biz blockIdx.z
#define bdx blockDim.x
#define bdy blockDim.y
#define bdz blockDim.z

namespace kernel {

template <typename Input>
__global__ void NaiveHistogram(Input in_data[], int out_freq[], int N, int symbols_per_thread);

/* Copied from J. Gomez-Luna et al */
template <typename T, typename FREQ>
__global__ void p2013Histogram(T*, FREQ*, size_t, int, int);

}  // namespace kernel

template <typename T>
__global__ void kernel::NaiveHistogram(T in_data[], int out_freq[], int N, int symbols_per_thread)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int j;
    if (i * symbols_per_thread < N) {  // if there is a symbol to count,
        for (j = i * symbols_per_thread; j < (i + 1) * symbols_per_thread; j++) {
            if (j < N) {
                unsigned int item = in_data[j];  // Symbol to count
                atomicAdd(&out_freq[item], 1);   // update bin count by 1
            }
        }
    }
}

template <typename T, typename FREQ>
__global__ void kernel::p2013Histogram(T* in_data, FREQ* out_freq, size_t N, int nbin, int R)
{
    // static_assert(
    //     std::numeric_limits<T>::is_integer and (not std::numeric_limits<T>::is_signed),
    //     "T must be `unsigned integer` type of {1,2,4} bytes");

    extern __shared__ int Hs[/*(nbin + 1) * R*/];

    const unsigned int warp_id     = (int)(tix / WARP_SIZE);
    const unsigned int lane        = tix % WARP_SIZE;
    const unsigned int warps_block = bdx / WARP_SIZE;
    const unsigned int off_rep     = (nbin + 1) * (tix % R);
    const unsigned int begin       = (N / warps_block) * warp_id + WARP_SIZE * blockIdx.x + lane;
    unsigned int       end         = (N / warps_block) * (warp_id + 1);
    const unsigned int step        = WARP_SIZE * gridDim.x;

    // final warp handles data outside of the warps_block partitions
    if (warp_id >= warps_block - 1) end = N;

    for (unsigned int pos = tix; pos < (nbin + 1) * R; pos += bdx) Hs[pos] = 0;
    __syncthreads();

    for (unsigned int i = begin; i < end; i += step) {
        int d = in_data[i];
        d     = d <= 0 and d >= nbin ? nbin / 2 : d;
        atomicAdd(&Hs[off_rep + d], 1);
    }
    __syncthreads();

    for (unsigned int pos = tix; pos < nbin; pos += bdx) {
        int sum = 0;
        for (int base = 0; base < (nbin + 1) * R; base += nbin + 1) { sum += Hs[base + pos]; }
        atomicAdd(out_freq + pos, sum);
    }
}

#endif /* D69BE972_2A8C_472E_930F_FFAB041F3F2B */
