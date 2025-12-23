#ifndef CUSZ_HEADER_H
#define CUSZ_HEADER_H

/**
 * @file header.h
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-01-22
 * (created) 2020-09-25, (rev.1) 2021-01-22 (rev.2) 2021-09-08 (rev.3)
 * 2022-02-26
 *
 * @copyright (C) 2020 by Washington State University, Argonne National
 * Laboratory See LICENSE in top-level directory
 *
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "cusz/type.h"

#define PSZHEADER_HEADER 0
#define PSZHEADER_ANCHOR 1
#define PSZHEADER_ENCODED 2
#define PSZHEADER_SPFMT 3
#define PSZHEADER_END 4

// @brief Memory alignment at 128 bytes for GPU if the archive is on device.
typedef struct psz_header {
  union {
    struct {
      psz_dtype dtype;
      psz_pipeline pipeline;
      psz_rc2 rc;

      // codec config (coarse-HF)
      int vle_sublen;
      int vle_pardeg;

      uint32_t entry[PSZHEADER_END + 1];  // segment entries

      // runtime sizes
      psz_len len;
      size_t splen;

      // internal loggin
      double user_input_eb;
      double min_val, max_val;

      // i/Hi
      INTERPOLATION_PARAMS intp_param;
    };

    // struct {
    //   uint8_t __[128];
    // };
  };
} psz_header;

psz_len pszheader_len(psz_header*);
size_t pszheader_len_linear(psz_header*);
size_t pszheader_segments(psz_header*);
size_t pszheader_filesize(psz_header*);
size_t pszheader_uncompressed_len(psz_header*);
size_t pszheader_compressed_bytes(psz_header*);

#ifdef __cplusplus
}
#endif

#endif
