/**
 * @file compressor.cc
 * @author Jiannan Tian
 * @brief cuSZ compressor of the default path
 * @version 0.3
 * @date 2023-06-02
 * (create) 2020-02-12
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory
 * @copyright (C) 2023 by Indiana University
 * See LICENSE in top-level directory
 *
 */

#include "compressor.hh"

#include "context.h"
#include "pipeline/compressor.inl"
#include "port.hh"
#include "tehm.hh"
#include "utils/config.hh"

// extra helper
namespace cusz {

int CompressorHelper::autotune_coarse_parhf(psz_context* ctx)
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
{
  auto tune_coarse_huffman_sublen = [](size_t len) {
    int current_dev = 0;
    GpuSetDevice(current_dev);
    GpuDeviceProp dev_prop{};
    GpuGetDeviceProperties(&dev_prop, current_dev);

    auto nSM = dev_prop.multiProcessorCount;
    auto allowed_block_dim = dev_prop.maxThreadsPerBlock;
    auto deflate_nthread =
        allowed_block_dim * nSM / HuffmanHelper::DEFLATE_CONSTANT;
    auto optimal_sublen = psz_utils::get_npart(len, deflate_nthread);
    optimal_sublen = psz_utils::get_npart(
                         optimal_sublen, HuffmanHelper::BLOCK_DIM_DEFLATE) *
                     HuffmanHelper::BLOCK_DIM_DEFLATE;

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
#elif defined(PSZ_USE_1API)
try {
  auto tune_coarse_huffman_sublen = [](size_t len) {
    int current_dev = 0;
    /*
    DPCT1093:0: The "current_dev" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    dpct::select_device(current_dev);
    dpct::device_info dev_prop{};
    dpct::dev_mgr::instance()
        .get_device(current_dev)
        .get_device_info(dev_prop);

    auto nEU = dev_prop.get_max_compute_units();
    auto allowed_block_dim = dev_prop.get_max_work_group_size();
    auto deflate_nthread =
        allowed_block_dim * nEU / HuffmanHelper::DEFLATE_CONSTANT;
    // Simple EU-SM conversion
    deflate_nthread /= 16;
    // SImple testing
    deflate_nthread /= 16;
    auto optimal_sublen = psz_utils::get_npart(len, deflate_nthread);
    optimal_sublen = psz_utils::get_npart(
                         optimal_sublen, HuffmanHelper::BLOCK_DIM_DEFLATE) *
                     HuffmanHelper::BLOCK_DIM_DEFLATE;

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
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#endif

}  // namespace cusz

using Ff4 = cusz::TEHM<float>;
using CFf4 = cusz::Compressor<Ff4>;

template class cusz::Compressor<Ff4>;
template CFf4* CFf4::init<psz_context>(psz_context* config, bool debug);
template CFf4* CFf4::init<psz_header>(psz_header* config, bool debug);
