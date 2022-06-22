/**
 * @file comp.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-05-01
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include <stdexcept>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include "component.hh"
#include "cusz.h"
#include "cusz/cc2c.h"

cusz_compressor* cusz_create(cusz_framework* framework, cusz_datatype type)
{
    return new cusz_compressor(framework, type);
}

cusz_error_status cusz_commit_space(cusz_compressor* comp, cusz_len const reserved_mem, cusz_framework* adjusted)
{
    if (adjusted)
        return comp->commit_space(reserved_mem, adjusted);
    else {
    }
    return CUSZ_SUCCESS;
}

cusz_error_status cusz_release(cusz_compressor* comp)
{
    delete comp;
    return CUSZ_SUCCESS;
}

cusz_error_status cusz_compress(
    cusz_compressor* comp,
    cusz_config*     config,
    void*            uncompressed,
    cusz_len const   uncomp_len,
    uint8_t**        compressed,
    size_t*          comp_bytes,
    cusz_header*     header,
    cusz_record**    record,
    cudaStream_t     stream)
{
    return comp->compress(config, uncompressed, uncomp_len, compressed, comp_bytes, header, record, stream);
}

cusz_error_status cusz_decompress(
    cusz_compressor* comp,
    cusz_header*     header,
    uint8_t*         compressed,
    size_t const     comp_len,
    void*            decompressed,
    cusz_len const   decomp_len,
    cusz_record**    record,
    cudaStream_t     stream)
{
    return comp->decompress(header, compressed, comp_len, decompressed, decomp_len, record, stream);
}