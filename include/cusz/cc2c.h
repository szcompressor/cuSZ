/**
 * @file cc2c.h
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-05-01
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_CC2C_H
#define CUSZ_CC2C_H

#include <cuda_runtime.h>
#include "../header.h"
#include "record.h"
#include "type.h"

// CC struct
struct cusz_compressor {
    void*           compressor;
    cusz_framework* framework;
    cusz_datatype   type;
    cusz_len        memlen{};
    cusz_len        datalen;
    void*           context;

    bool default_compressor{true};

    bool space_initialized{false};

    ~cusz_compressor() = default;
    cusz_compressor()  = default;
    cusz_compressor(cusz_framework*, cusz_datatype);
    cusz_compressor(const cusz_compressor&) = default;
    cusz_compressor& operator=(const cusz_compressor&) = default;
    cusz_compressor(cusz_compressor&&)                 = default;
    cusz_compressor& operator=(cusz_compressor&&) = default;

    /* Set size for memory allocation ; optionally adjust the framework */
    cusz_error_status commit_space(cusz_len const reserved_mem, cusz_framework* adjusted);

    /* Internally deep copy cusz_framework struct */
    cusz_error_status commit_framework(cusz_framework* framework);

    cusz_error_status compress(
        cusz_config*   config,
        void*          uncompressed,
        cusz_len const uncomp_len,
        uint8_t**      compressed,
        size_t*        comp_bytes,
        cusz_header*   header,
        void*          record,
        cudaStream_t   stream);

    cusz_error_status decompress(
        cusz_header*   header,
        uint8_t*       compressed,
        size_t const   comp_len,
        void*          decompressed,
        cusz_len const decomp_len,
        void*          record,
        cudaStream_t   stream);
};

#endif