/**
 * @file compressor.cu
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

#include "compressor.hh"
#include "framework.hh"
#include "pipeline/compressor_g.inl"
#include "utils/config.hh"

// extra helper
namespace cusz {

int CompressorHelper::autotune_coarse_parvle(cusz_context* ctx)
{
    auto tune_coarse_huffman_sublen = [](size_t len) {
        int current_dev = 0;
        cudaSetDevice(current_dev);
        cudaDeviceProp dev_prop{};
        cudaGetDeviceProperties(&dev_prop, current_dev);

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

template class cusz::Compressor<cusz::Framework<float>>;
