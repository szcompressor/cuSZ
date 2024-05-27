/**
 * @file hf_struct.c
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-06-03
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include "hf/hf_struct.h"
#include "layout.h"
#include "rt_config.h"

void hf_init_U4(
    hf_context*          ctx,
    hf_mem_layout*       mem,
    size_t const         datalen,
    int const            booklen,
    psz_device_property* prop,
    void*                ext_tmp_buf = nullptr)
{
    auto E = U4, H = U4, M = U4;

    int sublen, pardeg;
    psz_hf_tune_coarse_encoding(datalen, prop, &sublen, &pardeg);

    psz_memseg_assign(&mem->tmp, E, ext_tmp_buf != nullptr ? ext_tmp_buf : nullptr, datalen);
    psz_memseg_assign(&mem->book, H, nullptr, booklen);
    psz_memseg_assign(&mem->revbook, U1, nullptr, psz_hf_revbook_nbyte(booklen, 4));
    psz_memseg_assign(&mem->par_nbit, M, nullptr, ctx->pardeg);
    psz_memseg_assign(&mem->par_ncell, M, nullptr, ctx->pardeg);
    psz_memseg_assign(&mem->par_entry, M, nullptr, ctx->pardeg);
    psz_memseg_assign(&mem->bitstream, U4, nullptr, paz_hf_max_compressed_bytes(datalen));

    psz_memseg_assign(&mem->h_par_nbit, M, nullptr, ctx->pardeg);
    psz_memseg_assign(&mem->h_par_ncell, M, nullptr, ctx->pardeg);
    psz_memseg_assign(&mem->h_par_entry, M, nullptr, ctx->pardeg);

    psz_malloc_cuda(&mem->tmp);
    psz_malloc_cuda(&mem->book);
    psz_malloc_cuda(&mem->revbook);
    psz_malloc_cuda(&mem->par_nbit);
    psz_malloc_cuda(&mem->par_ncell);
    psz_malloc_cuda(&mem->par_entry);
    psz_malloc_cuda(&mem->out);

    psz_mallochost_cuda(&mem->h_par_nbit);
    psz_mallochost_cuda(&mem->h_par_ncell);
    psz_mallochost_cuda(&mem->h_par_entry);

    // auto ctx = new hf_context;

    ctx->book_desc      = new hf_book{nullptr, mem->book.buf, booklen};
    ctx->bitstream_desc = new hf_bitstream{
        mem->tmp.buf,
        mem->out.buf,
        new hf_chunk{mem->par_nbit.buf, mem->par_ncell.buf, mem->par_entry.buf},
        new hf_chunk{mem->h_par_nbit.buf, mem->h_par_ncell.buf, mem->h_par_entry.buf},
        sublen,
        pardeg,
        prop->sm_count};
}

void hf_free(hf_context* ctx, hf_mem_layout* mem)
{
    // free
    psz_free_cuda(&mem->tmp);
    psz_free_cuda(&mem->book);
    psz_free_cuda(&mem->revbook);
    psz_free_cuda(&mem->par_nbit);
    psz_free_cuda(&mem->par_ncell);
    psz_free_cuda(&mem->par_entry);
    psz_free_cuda(&mem->out);

    psz_freehost_cuda(&mem->h_par_nbit);
    psz_freehost_cuda(&mem->h_par_ncell);
    psz_freehost_cuda(&mem->h_par_entry);

    delete mem;

    delete ctx->bitstream_desc->d_metadata;
    delete ctx->bitstream_desc->h_metadata;
    delete ctx->bitstream_desc;
    delete ctx->book_desc;

    delete ctx;
}

#ifdef INLINE_MAIN

#include <unistd.h>

int main()
{
    auto ctx  = new hf_context;
    auto mem  = new hf_mem_layout;
    auto prop = new psz_device_property;

    psz_query_device(prop);

    hf_init_U4(ctx, mem, 512 * 512 * 128, 1 << 10, prop);

    sleep(3);

    hf_free(ctx, mem);

    return 0;
}

#endif