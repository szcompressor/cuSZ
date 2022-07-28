/**
 * @file cusz.h
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-29
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CUSZ_H
#define CUSZ_H

#include <stddef.h>

#include "cusz/cc2c.h"
#include "cusz/custom.h"
#include "cusz/record.h"
#include "cusz/type.h"
#include "header.h"

typedef struct cusz_compressor cusz_compressor;

cusz_compressor* cusz_create(cusz_framework* framework, cusz_datatype const type);

cusz_error_status cusz_commit_space(cusz_fixedlen_internal const* fixedlen, cusz_varlen_internal const* varlen);

cusz_error_status cusz_release(cusz_compressor* comp);

cusz_error_status cusz_compress(
    cusz_compressor* comp,
    cusz_config*     config,
    void*            uncompressed,
    cusz_len const   uncomp_len,
    uint8_t**        compressed,
    size_t*          comp_bytes,
    cusz_header*     header,
    void*            record,
    hipStream_t     stream);

cusz_error_status cusz_decompress(
    cusz_compressor* comp,
    cusz_header*     header,
    uint8_t*         compressed,
    size_t const     comp_len,
    void*            decompressed,
    cusz_len const   decomp_len,
    void*            record,
    hipStream_t     stream);

#endif

#ifdef __cplusplus
}
#endif
