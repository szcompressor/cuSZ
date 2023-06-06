/**
 * @file psz_lib.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-06-03
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include "psz_lib.h"
#include <cuda_runtime.h>
#include <cstdint>

#include "context.h"
#include "framework.hh"
#include "hf/hf.hh"
#include "kernel/lorenzo_all.hh"
#include "kernel/spv_gpu.hh"
#include "layout.h"
#include "rt_config.h"

// psz_error_status psz_make_context(psz_context** ctx) {}

// psz_error_status psz_init_F4_H4(psz_context* ctx, psz_mem_layout* mem)
// {
//     auto T = F4, E = U4, H = U4;

//     psz_memseg_assign(&mem->data, T, ctx->data_len);
//     psz_memseg_assign(&mem->errctrl, E, ctx->data_len);
//     psz_memseg_assign(&mem->errctrl, E, ctx->data_len);

//     return CUSZ_SUCCESS;
// }

psz_error_status psz_destroy() { return CUSZ_SUCCESS; }

namespace {
template <typename T>
T* ptr(psz_memseg& m)
{
    return (T*)m.buf;
}

template <typename T>
T* ptr(psz_memseg* m)
{
    return (T*)(m->buf);
}

template <psz_dtype T>
typename Ctype<T>::type* ptr(psz_memseg& m)
{
    return (typename Ctype<T>::type*)(m.buf);
}

}  // namespace

// when data is ready
psz_error_status psz_compress(
    int                  backend,
    psz_compressor*      comp,
    psz_device_property* prop,
    cusz_context*        config,
    void*                codec,
    psz_mem_layout*      mem,
    void*                in,
    size_t const         len,
    uint8_t*             out,
    size_t*              out_bytes,
    void*                stream = nullptr,
    size_t const         leny   = 1,
    size_t const         lenz   = 1,
    void*                record = nullptr)
{
    // if F4
    {
        using T = float;
        using E = uint32_t;
        using H = uint32_t;

        auto len_linearized = len * leny * lenz;
        auto len_freq       = config->radius * 2;

        compress_predict_lorenzo_i<T, E>(
            (T*)in, dim3(len, leny, lenz), config->eb, config->radius, ptr<E>(mem->errctrl), ptr<T>(mem->sp_val_full),
            ptr<E>(mem->sp_idx), nullptr /*nnz, reserved for next version*/, nullptr, (cudaStream_t)stream);

        psz_launch_p2013Histogram(
            prop, ptr<E>(mem->errctrl), len_linearized, ptr<U4>(mem->freq), len_freq, nullptr, (cudaStream_t)stream);

        auto hf = static_cast<cusz::HuffmanCodec<E, H>*>(codec);
        {
            hf->build_codebook(ptr<U4>(mem->freq), len_freq, (cudaStream_t)stream);

            auto hf_bitstream = ptr<U1>(mem->hf_bitstream);
            // auto dn_out_len = new size_t{mem->dn_out.len};

            hf->encode(
                ptr<E>(mem->errctrl), len_linearized, &hf_bitstream, &mem->hf_bitstream.len, (cudaStream_t)stream);
        }

        psz::spv_gather<T, uint32_t>(
            ptr<T>(mem->sp_val_full), len_linearized, ptr<T>(mem->sp_val), ptr<U4>(mem->sp_idx), &mem->nnz, nullptr,
            (cudaStream_t)stream);
    }

    // if F8
    {
    }

    return CUSZ_SUCCESS;
}

psz_error_status psz_decompress(
    psz_compressor* comp,
    psz_header*     header,
    uint8_t*        compressed,
    size_t const    comp_len,
    psz_mem_layout* mem,
    void*           codec,
    void*           decompressed,
    psz_len const   decomp_len,
    void*           stream = nullptr,
    void*           record = nullptr)
{
    {  // if F4
        using T = float;
        using M = uint32_t;
        using E = uint32_t;
        using H = uint32_t;

        auto at = [&](auto SEG) { return compressed + header->entry[SEG]; };

        psz::spv_scatter<T, uint32_t>(
            (T*)at(psz_header::SP_VAL), (M*)at(psz_header::SP_IDX), header->nnz, (T*)decompressed, nullptr,
            (cudaStream_t)stream);

        auto hf = static_cast<cusz::HuffmanCodec<E, H>*>(codec);
        hf->decode(at(psz_header::VLE), ptr<E>(mem->errctrl));

        decompress_predict_lorenzo_i<T, E>(
            ptr<E>(mem->errctrl), dim3(header->x, header->y, header->z), (T*)decompressed, nullptr, 0, header->eb,
            header->radius, (T*)decompressed, nullptr, (cudaStream_t)stream);
    }

    // if F8
    {
    }

    return CUSZ_SUCCESS;
}
