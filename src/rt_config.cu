/**
 * @file rt_config.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-06-03
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include "busyheader.hh"
#include "cusz/type.h"
#include "kernel/detail/hist_cuda.inl"
#include "rt_config.h"
#include "utils/timer.hh"

// query
psz_error_status psz_query_device(psz_device_property* prop)
{
    int device_id;
    cudaGetDevice(&device_id);

    // cout << "device_id\t" << device_id << endl;
    // cudaSetDevice(device_id);

    cudaDeviceProp dev_prop{};
    cudaGetDeviceProperties(&dev_prop, device_id);

    // for embarassingly parallelized (coarse-grain) VLE (Huffman)
    {
        prop->sm_count     = dev_prop.multiProcessorCount;
        prop->max_blockdim = dev_prop.maxThreadsPerBlock;
    }

    // for fin-grain Histogram (p2013Histogram)
    {
        // Maximum shared memory available per block in bytes
        cudaDeviceGetAttribute(&prop->max_shmem_bytes, cudaDevAttrMaxSharedMemoryPerBlock, device_id);

        // The maximum optin shared memory per block. This value may vary by chip
        cudaDeviceGetAttribute(&prop->max_shmem_bytes_opt_in, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
    }

    return CUSZ_SUCCESS;
}

psz_error_status psz_launch_p2013Histogram(
    psz_device_property* prop,
    uint32_t*            in_data,
    size_t const         in_len,
    uint32_t*            out_freq,
    int const            num_buckets,
    float*               milliseconds,
    void*                stream)
{
    auto max_bytes = max(prop->max_shmem_bytes, prop->max_shmem_bytes_opt_in);

    cudaFuncSetAttribute(
        kernel::p2013Histogram<uint32_t, uint32_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_bytes);

    int items_per_thread, r_per_block, grid_dim, block_dim, shmem_use;
    {
        items_per_thread = 1;
        r_per_block      = (max_bytes / sizeof(int)) / (num_buckets + 1);
        grid_dim         = prop->sm_count;
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
    }

    // start timing
    CREATE_GPUEVENT_PAIR;
    START_GPUEVENT_RECORDING((cudaStream_t)stream);

    kernel::p2013Histogram<<<grid_dim, block_dim, shmem_use, (cudaStream_t)stream>>>  //
        (in_data, out_freq, in_len, num_buckets, r_per_block);

    STOP_GPUEVENT_RECORDING((cudaStream_t)stream);

    cudaStreamSynchronize((cudaStream_t)stream);
    TIME_ELAPSED_GPUEVENT(milliseconds);
    DESTROY_GPUEVENT_PAIR;

    return CUSZ_SUCCESS;
}

psz_error_status psz_hf_tune_coarse_encoding(size_t const len, psz_device_property* prop, int* sublen, int* pardeg)
{
    // static const int BLOCK_DIM_ENCODE  = 256;
    static const int BLOCK_DIM_DEFLATE = 256;
    static const int DEFLATE_CONSTANT  = 4;

    auto div = [](auto whole, auto part) -> uint32_t {
        if (whole == 0) throw std::runtime_error("Dividend is zero.");
        if (part == 0) throw std::runtime_error("Divisor is zero.");
        return (whole - 1) / part + 1;
    };

    auto nthread = prop->max_blockdim * prop->sm_count / DEFLATE_CONSTANT;
    auto _       = div(len, nthread);
    *sublen      = div(_, BLOCK_DIM_DEFLATE) * BLOCK_DIM_DEFLATE;
    *pardeg      = div(len, *sublen);

    return CUSZ_SUCCESS;
}

int psz_hf_revbook_nbyte(int booklen, int symbol_bytewidth)
{
    // symbol_bytewidth === sizeof(H) in other contexts
    return symbol_bytewidth * (2 * (symbol_bytewidth * 8)) + symbol_bytewidth * booklen;
}

size_t paz_hf_max_compressed_bytes(size_t datalen) { return datalen / 2; }