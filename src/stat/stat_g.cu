/**
 * @file stat_g.cu
 * @author Cody Rivera, Jiannan Tian
 * @brief Fast histogramming from [Gómez-Luna et al. 2013], wrapper
 * @version 0.3
 * @date 2022-11-02
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "../kernel/detail/hist.inl"

#include "cusz/type.h"
#include "stat/stat.h"
#include "stat/stat_g.hh"

template <typename T>
cusz_error_status asz::stat::histogram(
    T*           in_data,
    size_t const in_len,
    uint32_t*    out_freq,
    int const    num_buckets,
    float*       milliseconds,
    cudaStream_t stream)
{
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

    CREATE_CUDAEVENT_PAIR;
    START_CUDAEVENT_RECORDING(stream);

    kernel::p2013Histogram<<<grid_dim, block_dim, shmem_use, stream>>>  //
        (in_data, out_freq, in_len, num_buckets, r_per_block);

    STOP_CUDAEVENT_RECORDING(stream);

    cudaStreamSynchronize(stream);
    TIME_ELAPSED_CUDAEVENT(milliseconds);
    DESTROY_CUDAEVENT_PAIR;

    return CUSZ_SUCCESS;
}

#define INIT_HIST_AND_C(Tname, T)                                                                                     \
    template cusz_error_status asz::stat::histogram<T>(T*, size_t const, uint32_t*, int const, float*, cudaStream_t); \
                                                                                                                      \
    cusz_error_status histogram_T##Tname(                                                                             \
        T* in_data, size_t const in_len, uint32_t* out_freq, int const num_buckets, float* milliseconds,              \
        cudaStream_t stream)                                                                                          \
    {                                                                                                                 \
        return asz::stat::histogram<T>(in_data, in_len, out_freq, num_buckets, milliseconds, stream);                 \
    }

INIT_HIST_AND_C(ui8, uint8_t)
INIT_HIST_AND_C(ui16, uint16_t)
INIT_HIST_AND_C(ui32, uint32_t)
INIT_HIST_AND_C(ui64, uint64_t)

#undef INIT_HIST_AND_C