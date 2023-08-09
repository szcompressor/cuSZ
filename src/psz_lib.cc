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
#include "kernel/l23.hh"
#include "kernel/spv_gpu.hh"
#include "rt_config.h"
#include "utils2/layout.h"

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
T* ptr(pszmem& m)
{
    return (T*)m.buf;
}

template <typename T>
T* ptr(pszmem* m)
{
    return (T*)(m->buf);
}

template <psz_dtype T>
typename Ctype<T>::type* ptr(pszmem& m)
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
    pszmem_pool*         mem,
    void*                in,
    size_t const         len,
    uint8_t*             out,
    size_t*              out_bytes,
    void*                stream = nullptr,
    size_t const         leny   = 1,
    size_t const         lenz   = 1,
    void*                record = nullptr)
{
    return CUSZ_SUCCESS;
}

psz_error_status psz_decompress(
    psz_compressor* comp,
    psz_header*     header,
    uint8_t*        compressed,
    size_t const    comp_len,
    pszmem_pool*    mem,
    void*           codec,
    void*           decompressed,
    psz_len const   decomp_len,
    void*           stream = nullptr,
    void*           record = nullptr)
{
    return CUSZ_SUCCESS;
}
