/**
 * @file autotune.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-03-08
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef UTILS_AUTOTUNE_CUH
#define UTILS_AUTOTUNE_CUH

#include <cuda_runtime.h>
#include "../common/configs.hh"
#include "../context.hh"

struct AutoconfigHelper {
    static int autotune(cuszCTX* ctx)
    {
        auto tune_coarse_huffman_sublen = [](size_t len) {
            int current_dev = 0;
            cudaSetDevice(current_dev);
            cudaDeviceProp dev_prop{};
            cudaGetDeviceProperties(&dev_prop, current_dev);

            auto nSM               = dev_prop.multiProcessorCount;
            auto allowed_block_dim = dev_prop.maxThreadsPerBlock;
            auto deflate_nthread   = allowed_block_dim * nSM / HuffmanHelper::DEFLATE_CONSTANT;
            auto optimal_sublen    = ConfigHelper::get_npart(len, deflate_nthread);
            optimal_sublen         = ConfigHelper::get_npart(optimal_sublen, HuffmanHelper::BLOCK_DIM_DEFLATE) *
                             HuffmanHelper::BLOCK_DIM_DEFLATE;

            return optimal_sublen;
        };

        auto get_coarse_pardeg = [&](size_t len, int& sublen, int& pardeg) {
            sublen = tune_coarse_huffman_sublen(len);
            pardeg = ConfigHelper::get_npart(len, sublen);
        };

        // TODO should be move to somewhere else, e.g., cusz::par_optmizer
        if (ctx->use.autotune_vle_pardeg)
            get_coarse_pardeg(ctx->data_len, ctx->vle_sublen, ctx->vle_pardeg);
        else
            ctx->vle_pardeg = ConfigHelper::get_npart(ctx->data_len, ctx->vle_sublen);

        return ctx->vle_pardeg;
    }
};

#endif
