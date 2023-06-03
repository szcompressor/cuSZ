/**
 * @file layout.c
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-06-03
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include "layout.h"
#include <cuda_runtime.h>
#include <cstring>

void psz_memseg_assign(psz_memseg* m, psz_dtype const t, void* b, uint32_t const l)
{
    m->type = t, m->buf = b, m->len = l;
}

void psz_malloc_cuda(psz_memseg* m)
{
    auto bytes = m->len * (((int)m->type) % 10);
    if (m->buf == nullptr) cudaMalloc(&m->buf, bytes);
    cudaMemset(m->buf, 0x0, bytes);
}

void psz_mallochost_cuda(psz_memseg* m)
{
    auto bytes = m->len * (((int)m->type) % 10);
    if (m->buf == nullptr) cudaMallocHost(&m->buf, bytes);
    memset(m->buf, 0x0, bytes);
}

void psz_mallocmanaged_cuda(psz_memseg* m)
{
    auto bytes = m->len * (((int)m->type) % 10);
    if (m->buf == nullptr) cudaMallocManaged(&m->buf, bytes);
    memset(m->buf, 0x0, bytes);
}

void psz_free_cuda(psz_memseg* m)
{
    if (m->buf) cudaFree(m->buf);
}

void psz_freehost_cuda(psz_memseg* m)
{
    if (m->buf) cudaFreeHost(m->buf);
}