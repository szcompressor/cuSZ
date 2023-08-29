/**
 * @file compressor.cc
 * @author Jiannan Tian
 * @brief cuSZ compressor of the default path
 * @version 0.3
 * @date 2023-06-02
 * (create) 2020-02-12
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * @copyright (C) 2023 by Indiana University
 * See LICENSE in top-level directory
 *
 */

#include "port.hh"
#include "compressor.hh"
#include "context.h"
#include "tehm.hh"
#include "pipeline/compressor.inl"
#include "utils/config.hh"

// extra helper
namespace cusz {

int CompressorHelper::autotune_coarse_parhf(cusz_context* ctx)
{
    auto tune_coarse_huffman_sublen = [](size_t len) {
        int current_dev = 0;
        GpuSetDevice(current_dev);
        GpuDeviceProp dev_prop{};
        GpuGetDeviceProperties(&dev_prop, current_dev);

        auto nSM               = dev_prop.multiProcessorCount;
        auto allowed_block_dim = dev_prop.maxThreadsPerBlock;
        auto deflate_nthread   = allowed_block_dim * nSM / HuffmanHelper::DEFLATE_CONSTANT;
        auto optimal_sublen    = psz_utils::get_npart(len, deflate_nthread);
        optimal_sublen =
            psz_utils::get_npart(optimal_sublen, HuffmanHelper::BLOCK_DIM_DEFLATE) * HuffmanHelper::BLOCK_DIM_DEFLATE;

        return optimal_sublen;
    };

    auto get_coarse_pardeg = [&](size_t len, int& sublen, int& pardeg) {
        sublen = tune_coarse_huffman_sublen(len);
        pardeg = psz_utils::get_npart(len, sublen);
    };

    // TODO should be move to somewhere else, e.g., cusz::par_optmizer
    if (ctx->use_autotune_hf)
        get_coarse_pardeg(ctx->data_len, ctx->vle_sublen, ctx->vle_pardeg);
    else
        ctx->vle_pardeg = psz_utils::get_npart(ctx->data_len, ctx->vle_sublen);

    return ctx->vle_pardeg;
}

}  // namespace cusz

using Ff4 = cusz::TEHM<float>;
using CFf4 = cusz::Compressor<Ff4>;

template class cusz::Compressor<Ff4>;
template CFf4* CFf4::init<cusz_context>(cusz_context* config, bool debug);
template CFf4* CFf4::init<cusz_header>(cusz_header* config, bool debug);
