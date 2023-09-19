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

#include "cusz/type.h"
#include "hf/hf2.h"
#include "hf/hf_bk.hh"
#include "hf/hfcodec.cu.hh"
#include "hf/hf_struct.h"
#include "typing.hh"

template <typename T, typename H, typename M = uint32_t>
psz_error_status hf_encode(
    hf_context* ctx, hfmem_pool* mem, T* d_in, size_t const inlen,
    uint8_t** out, size_t* outlen, void* stream)
{
  float _time_book;
  float time_lossless;
  *out = (uint8_t*)mem->bitstream.buf;

  psz::hf_buildbook<CUDA, T, H>(
      ctx->freq, ctx->booklen, (H*)mem->book.buf, (uint8_t*)mem->revbook.buf,
      mem->revbook.len, &_time_book, stream);

  psz::hf_encode_coarse_rev2<T, H, M>(
      d_in, inlen, ctx->book_desc, ctx->bitstream_desc, &ctx->total_nbit,
      &ctx->total_ncell, &time_lossless, (cudaStream_t)stream);

  // TODO memcpy on event

  return CUSZ_SUCCESS;
}

template <typename T, typename H, typename M = uint32_t>
psz_error_status hf_decode(
    hf_context* ctx, hfmem_pool* mem, uint8_t* in, T* out, void* stream)
{
  psz::hf_decode_coarse<T, H, M>(
      mem->bitstream.buf, mem->revbook.buf, mem->revbook.len,
      mem->par_nbit.buf, mem->par_entry.buf,
      ctx->sublen /* TODO check ctx assignment */, ctx->pardeg, out,
      ctx->time_lossless, stream);

  return CUSZ_SUCCESS;
}