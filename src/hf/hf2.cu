/**
 * @file hf2.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-06-04
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <numeric>

#include "common/type_traits.hh"
#include "cusz/type.h"
#include "hf/hf2.h"
#include "hf/hf_bookg.hh"
#include "hf/hf_codecg.hh"
#include "hf/hf_struct.h"

template <typename T, typename H, typename M = uint32_t>
psz_error_status
hf_encode(hf_context* ctx, hf_mem_layout* mem, T* d_in, size_t const inlen, uint8_t** out, size_t* outlen, void* stream)
{
    float time_book;
    float time_lossless;
    *out = (uint8_t*)mem->bitstream.buf;

    psz::hf_buildbook_g<T, H>(
        ctx->freq, ctx->booklen, (H*)mem->book.buf, (uint8_t*)mem->revbook.buf, mem->revbook.len, &time_book,
        (cudaStream_t)stream);

    psz::hf_encode_coarse_rev1<T, H, M>(
        d_in, inlen, ctx->book_desc, ctx->bitstream_desc, (uint8_t*)mem->bitstream.buf, outlen, &time_lossless,
        (cudaStream_t)stream);

    // auxillary, put in ctx rather than export as part of output
    ctx->total_nbit = std::accumulate(
        (M*)ctx->bitstream_desc->h_metadata->bits,
        (M*)ctx->bitstream_desc->h_metadata->bits + ctx->bitstream_desc->pardeg, (size_t)0);
    ctx->total_ncell = std::accumulate(
        (M*)ctx->bitstream_desc->h_metadata->cells,
        (M*)ctx->bitstream_desc->h_metadata->cells + ctx->bitstream_desc->pardeg, (size_t)0);

    return CUSZ_SUCCESS;
}

template <typename T, typename H, typename M = uint32_t>
psz_error_status hf_decode(hf_context* ctx, hf_mem_layout* mem, uint8_t* in, T* out, void* stream)
{
    psz::hf_decode_coarse<T, H, M>(
        mem->bitstream.buf, mem->revbook.buf, mem->revbook.len, mem->par_nbit.buf, mem->par_entry.buf,
        ctx->sublen /* TODO check ctx assignment */, ctx->pardeg, out, ctx->time_lossless, stream);

    return CUSZ_SUCCESS;
}