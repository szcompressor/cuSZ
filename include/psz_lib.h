/**
 * @file psz_lib.h
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-06-03
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef CAF52C2D_DD3A_42DC_9F1F_EA202D2A79D5
#define CAF52C2D_DD3A_42DC_9F1F_EA202D2A79D5

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

#include "cusz/custom.h"
#include "cusz/record.h"
#include "cusz/type.h"
#include "header.h"
#include "utils2/layout.h"

typedef cusz_error_status psz_error_status;
typedef cusz_framework    psz_framework;
typedef cusz_compressor   psz_compressor;
typedef cusz_datatype     psz_datatype;
typedef cusz_len          psz_len;
typedef cusz_config       psz_config;
// typedef cusz_context      psz_context;

psz_compressor* psz_init(psz_framework* framework, psz_datatype const type);

psz_error_status psz_free(psz_compressor* comp);

psz_error_status psz_compress(
    psz_compressor* comp,
    psz_config*     config,
    void*           uncompressed,
    psz_len const   uncomp_len,
    uint8_t**       compressed,
    size_t*         comp_bytes,
    psz_header*     header,
    void*           record,
    void*           stream);

psz_error_status psz_decompress(
    psz_compressor* comp,
    psz_header*     header,
    uint8_t*        compressed,
    size_t const    comp_len,
    void*           decompressed,
    psz_len const   decomp_len,
    void*           record,
    void*           stream);

#ifdef __cplusplus
}
#endif

#endif /* CAF52C2D_DD3A_42DC_9F1F_EA202D2A79D5 */
