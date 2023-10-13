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

#include "cusz/type.h"
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct alignas(128) psz_header {
  static const int HEADER = 0;
  static const int ANCHOR = 1;
  static const int VLE = 2;
  static const int SPFMT = 3;
  static const int END = 4;

  // uint32_t self_bytes : 16;
  // uint32_t fp : 1;
  uint32_t byte_vle : 4;           // 4, 8
  uint32_t nz_density_factor : 8;  // TODO configurate it
  uint32_t codecs_in_use : 2;
  uint32_t vle_pardeg;
  uint32_t x, y, z, w;
  psz_dtype dtype;
  double eb;
  uint32_t radius : 16;
  int splen;

  uint32_t entry[END + 1];

  psz_predtype pred_type;

  // uint32_t byte_uncomp : 4;   // T; 1, 2, 4, 8
  // uint32_t byte_errctrl : 3;  // 1, 2, 4
  // uint32_t byte_meta : 4;     // 4, 8
  
  
  INTERPOLATION_PARAMS intp_param;
} psz_header;
typedef psz_header pszheader;

#ifdef __cplusplus
}
#endif

#endif
