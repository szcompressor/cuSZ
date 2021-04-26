/**
 * @file hist.cuh
 * @author Jiannan Tian, Cody Rivera (cjrivera1@crimson.ua.edu)
 * @brief
 * @version 0.2
 * @date 2021-04-06
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

// TODO histogram orignal license

#ifndef KERNEL_HIST_CUH
#define KERNEL_HIST_CUH

#include <math.h>
#include <stdlib.h>
#include "../type_aliasing.hh"

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

static const auto WARP_SIZE = 32;

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

namespace sc21 {

// TODO cpp17 static-if compilation
template <typename Input, typename Output_UInt, bool CutOff = false>
__global__ void Histogram(Input* input_data, Output_UInt* output, size_t N, int num_syms, int r_per_block)
{
    // static_assert(
    //     (std::is_same<Input, UI1>::value or std::is_same<Input, UI2>::value or std::is_same<Input, UI4>::value)  //
    //         or ((std::is_same<Input, FP4>::value or std::is_same<Input, FP8>::value) and (CutOff == true)),
    //     "When Input is FP-type, CutOff must be enabled.");

    extern __shared__ int Hs[/*(num_syms + 1) * r_per_block*/];

    const unsigned int warp_id     = (int)(tix / WARP_SIZE);
    const unsigned int lane        = tix % WARP_SIZE;
    const unsigned int warps_block = bdx / WARP_SIZE;
    const unsigned int off_rep     = (num_syms + 1) * (tix % r_per_block);
    const unsigned int begin       = (N / warps_block) * warp_id + WARP_SIZE * blockIdx.x + lane;
    unsigned int       end         = (N / warps_block) * (warp_id + 1);
    const unsigned int step        = WARP_SIZE * gridDim.x;

    // final warp handles data outside of the warps_block partitions
    if (warp_id >= warps_block - 1) end = N;

    for (unsigned int pos = tix; pos < (num_syms + 1) * r_per_block; pos += bdx) Hs[pos] = 0;
    __syncthreads();

    /* template static branches */
    // if CONSTEXPR ((std::is_same<Input, FP4>::value or std::is_same<Input, FP8>::value) and CutOff == true) {
    //     /* post quant (dryrun) is done here */
    //     auto radius = num_syms / 2;
    //     for (unsigned int i = begin; i < end; i += step) {
    //         int d = static_cast<int>(input_data[i]);
    //         d     = fabs(d) >= radius ? 0 : d;
    //         atomicAdd(&Hs[off_rep + d + radius], 1);
    //     }
    // }
    // else {
    /* original simplistic histogram */
    for (unsigned int i = begin; i < end; i += step) {
        int d = input_data[i];
        atomicAdd(&Hs[off_rep + d], 1);
    }
    // }
    __syncthreads();

    for (unsigned int pos = tix; pos < num_syms; pos += bdx) {
        int sum = 0;
        for (int base = 0; base < (num_syms + 1) * r_per_block; base += num_syms + 1) { sum += Hs[base + pos]; }
        atomicAdd(output + pos, sum);
    }
}

template <typename Input, typename Output_UInt, bool CutOff = false>
__global__ void
HistogramFPQuantCandidate(Input* input_data, Output_UInt* output, size_t N, int num_syms, int r_per_block)
{
    // static_assert(
    //     (std::is_same<Input, UI1>::value or std::is_same<Input, UI2>::value or std::is_same<Input, UI4>::value)  //
    //         or ((std::is_same<Input, FP4>::value or std::is_same<Input, FP8>::value) and (CutOff == true)),
    //     "When Input is FP-type, CutOff must be enabled.");

    extern __shared__ int Hs[/*(num_syms + 1) * r_per_block*/];

    const unsigned int warp_id     = (int)(tix / WARP_SIZE);
    const unsigned int lane        = tix % WARP_SIZE;
    const unsigned int warps_block = bdx / WARP_SIZE;
    const unsigned int off_rep     = (num_syms + 1) * (tix % r_per_block);
    const unsigned int begin       = (N / warps_block) * warp_id + WARP_SIZE * blockIdx.x + lane;
    unsigned int       end         = (N / warps_block) * (warp_id + 1);
    const unsigned int step        = WARP_SIZE * gridDim.x;

    // final warp handles data outside of the warps_block partitions
    if (warp_id >= warps_block - 1) end = N;

    for (unsigned int pos = tix; pos < (num_syms + 1) * r_per_block; pos += bdx) Hs[pos] = 0;
    __syncthreads();

    /* template static branches */
    // if CONSTEXPR ((std::is_same<Input, FP4>::value or std::is_same<Input, FP8>::value) and CutOff == true) {
    /* post quant (dryrun) is done here */
    auto radius = num_syms / 2;
    for (unsigned int i = begin; i < end; i += step) {
        int d = static_cast<int>(input_data[i]);
        d     = fabs(d) >= radius ? 0 : d;
        atomicAdd(&Hs[off_rep + d + radius], 1);
    }
    // }
    // else {
    //     /* original simplistic histogram */
    //     for (unsigned int i = begin; i < end; i += step) {
    //         int d = input_data[i];
    //         atomicAdd(&Hs[off_rep + d], 1);
    //     }
    // }
    __syncthreads();

    for (unsigned int pos = tix; pos < num_syms; pos += bdx) {
        int sum = 0;
        for (int base = 0; base < (num_syms + 1) * r_per_block; base += num_syms + 1) { sum += Hs[base + pos]; }
        atomicAdd(output + pos, sum);
    }
}

}  // namespace sc21

#endif