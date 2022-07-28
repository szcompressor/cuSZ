/**
 * @file compressor.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-23
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */


#include <hip/hip_runtime.h>
#include "compressor.hh"
#include "common/configs.hh"
#include "framework.hh"

namespace cusz {

template <class B>
Compressor<B>::~Compressor()
{
    pimpl.reset();
}

template <class B>
Compressor<B>::Compressor() : pimpl{std::make_unique<impl>()}
{
}

template <class B>
Compressor<B>::Compressor(const Compressor<B>& old) : pimpl{std::make_unique<impl>(*old.pimpl)}
{
}

template <class B>
Compressor<B>& Compressor<B>::operator=(const Compressor<B>& old)
{
    *pimpl = *old.pimpl;
    return *this;
}

template <class B>
Compressor<B>::Compressor(Compressor<B>&&) = default;

template <class B>
Compressor<B>& Compressor<B>::operator=(Compressor<B>&&) = default;

//------------------------------------------------------------------------------

template <class B>
void Compressor<B>::init(Context* config, bool dbg_print)
{
    pimpl->init(config, dbg_print);
}

template <class B>
void Compressor<B>::init(Header* config, bool dbg_print)
{
    pimpl->init(config, dbg_print);
}

template <class B>
void Compressor<B>::compress(
    Context*          config,
    Compressor<B>::T* uncompressed,
    BYTE*&            compressed,
    size_t&           compressed_len,
    hipStream_t      stream,
    bool              dbg_print)
{
    pimpl->compress(config, uncompressed, compressed, compressed_len, stream, dbg_print);
}

template <class B>
void Compressor<B>::decompress(
    Header*           config,
    BYTE*             compressed,
    Compressor<B>::T* decompressed,
    hipStream_t      stream,
    bool              dbg_print)
{
    pimpl->decompress(config, compressed, decompressed, stream, dbg_print);
}

template <class B>
void Compressor<B>::clear_buffer()
{
    pimpl->clear_buffer();
}

// getter

template <class B>
void Compressor<B>::export_header(Header& header)
{
    pimpl->export_header(header);
}

template <class B>
void Compressor<B>::export_header(Header* header)
{
    pimpl->export_header(header);
}

template <class B>
void Compressor<B>::export_timerecord(TimeRecord* ext_timerecord)
{
    pimpl->export_timerecord(ext_timerecord);
}

}  // namespace cusz

// extra helper
namespace cusz {

int CompressorHelper::autotune_coarse_parvle(Context* ctx)
{
    auto tune_coarse_huffman_sublen = [](size_t len) {
        int current_dev = 0;
        hipSetDevice(current_dev);
        hipDeviceProp_t dev_prop{};
        hipGetDeviceProperties(&dev_prop, current_dev);

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

}  // namespace cusz

template class cusz::Compressor<cusz::PredefinedCombination<float>::LorenzoFeatured>;
// template class cusz::Compressor<cusz::PredefinedCombination<float>::Spline3Featured>;
