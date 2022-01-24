/**
 * @file hist.cuh
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

#ifndef CUSZ_KERNEL_HIST_CUH
#define CUSZ_KERNEL_HIST_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <limits>

#include "../common.hh"
#include "../utils/timer.hh"

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

namespace kernel_wrapper {

/**
 * @brief Get frequency: a kernel wrapper
 *
 * @tparam T input type
 * @param in_data input device array
 * @param in_len input host var; len of in_data
 * @param out_freq output device array
 * @param nbin input host var; len of out_freq
 * @param milliseconds output time elapsed
 * @param stream optional stream
 */
template <typename T>
void get_frequency(
    T*           in_data,
    size_t       in_len,
    cusz::FREQ*  out_freq,
    int          nbin,
    float&       milliseconds,
    cudaStream_t stream = nullptr);

}  // namespace kernel_wrapper

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
    static_assert(
        std::numeric_limits<T>::is_integer and (not std::numeric_limits<T>::is_signed),
        "T must be `unsigned integer` type of {1,2,4} bytes");

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
        atomicAdd(&Hs[off_rep + d], 1);
    }
    __syncthreads();

    for (unsigned int pos = tix; pos < nbin; pos += bdx) {
        int sum = 0;
        for (int base = 0; base < (nbin + 1) * R; base += nbin + 1) { sum += Hs[base + pos]; }
        atomicAdd(out_freq + pos, sum);
    }
}

template <typename T>
void kernel_wrapper::get_frequency(
    T*           in_data,
    size_t       in_len,
    cusz::FREQ*  out_freq,
    int          num_buckets,
    float&       milliseconds,
    cudaStream_t stream)
{
    static_assert(
        std::numeric_limits<T>::is_integer and (not std::numeric_limits<T>::is_signed),
        "To get frequency, `T` must be unsigned integer type of {1,2,4} bytes");

    int device_id, max_bytes, num_SMs;
    int items_per_thread, r_per_block, grid_dim, block_dim, shmem_use;

    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, device_id);

    auto query_maxbytes = [&]() {
        int max_bytes_opt_in;
        cudaDeviceGetAttribute(&max_bytes, cudaDevAttrMaxSharedMemoryPerBlock, device_id);

        // account for opt-in extra shared memory on certain architectures
        cudaDeviceGetAttribute(&max_bytes_opt_in, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
        max_bytes = std::max(max_bytes, max_bytes_opt_in);

        // config kernel attribute
        cudaFuncSetAttribute(
            kernel::p2013Histogram<T, cusz::FREQ>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_bytes);
    };

    auto optimize_launch = [&]() {
        items_per_thread = 1;
        r_per_block      = (max_bytes / sizeof(int)) / (num_buckets + 1);
        grid_dim         = num_SMs;
        // fits to size
        block_dim = ((((in_len / (grid_dim * items_per_thread)) + 1) / 64) + 1) * 64;
        while (block_dim > 1024) {
            if (r_per_block <= 1) { block_dim = 1024; }
            else {
                r_per_block /= 2;
                grid_dim *= 2;
                block_dim = ((((in_len / (grid_dim * items_per_thread)) + 1) / 64) + 1) * 64;
            }
        }
        shmem_use = ((num_buckets + 1) * r_per_block) * sizeof(int);
    };

    query_maxbytes();
    optimize_launch();

    cuda_timer_t t;
    t.timer_start(stream);
    kernel::p2013Histogram<<<grid_dim, block_dim, shmem_use, stream>>>  //
        (in_data, out_freq, in_len, num_buckets, r_per_block);
    t.timer_end(stream);
    cudaStreamSynchronize(stream);

    milliseconds = t.get_time_elapsed();
}

#endif
