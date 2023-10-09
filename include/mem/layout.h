/**
 * @file layout.h
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-06-03
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef DA377C1A_D4A3_492C_A9E1_44072067050A
#define DA377C1A_D4A3_492C_A9E1_44072067050A

#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#include "cusz/type.h"
#include "mem/memseg.h"

typedef enum pszmem_runtime_type {
  PszHeader = 0,
  PszQuant = 1,
  PszHist = 2,
  PszSpVal = 3,
  PszSpIdx = 4,
  PszArchive = 5,
  PszHf______ = 6,  // hf dummy start
  pszhf_header = 7,
  PszHfBook = 8,
  PszHfRevbook = 9,
  PszHfParNbit = 10,
  PszHfParNcell = 11,
  PszHfParEntry = 12,
  PszHfBitstream = 13,
  PszHfArchive = 14,
  END
} pszmem_runtime_type;
// use scenario: dump intermediate dat
typedef pszmem_runtime_type pszmem_dump;


typedef struct psz_memory_pool {
  size_t seg_len[END];
  size_t seg_entry[END];
  float density{0.2};
  size_t compressed_len;
  int nnz;

  pszmem __pool;
  pszmem data;
  pszmem ectrl;
  pszmem spval;
  pszmem spidx;
  pszmem hf_bitstream;
  pszmem anchor;
  pszmem freq;
} pszmem_pool;

// void init(pszmem_pool*, size_t);
// void destroy(pszmem_pool*);

#ifdef __cplusplus
}
#endif

#endif /* DA377C1A_D4A3_492C_A9E1_44072067050A */
