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
#include "utils2/memseg.h"

typedef enum pszmem_runtime_type {
    PszHeader      = 0,
    PszQuant       = 1,
    PszHist        = 2,
    PszSpVal       = 3,
    PszSpIdx       = 4,
    PszArchive     = 5,
    PszHf______    = 6,  // hf dummy start
    PszHfHeader    = 7,
    PszHfBook      = 8,
    PszHfRevbook   = 9,
    PszHfParNbit   = 10,
    PszHfParNcell  = 11,
    PszHfParEntry  = 12,
    PszHfBitstream = 13,
    PszHfArchive   = 14,
    END
} pszmem_runtime_type;
// use scenario: dump intermediate dat
typedef pszmem_runtime_type pszmem_dump;

// TODO move to header.h
typedef struct alignas(128) psz_header {
    static const int HEADER = 0;
    static const int ANCHOR = 1;
    static const int VLE    = 2;
    static const int SP_VAL = 3;
    static const int SP_IDX = 4;
    static const int END    = 5;

    // uint32_t self_bytes : 16;
    // uint32_t fp : 1;
    // uint32_t byte_vle : 4;  // 4, 8
    // uint32_t nz_density_factor : 8;
    // uint32_t codecs_in_use : 2;
    // uint32_t vle_pardeg;
    uint32_t x, y, z, w;
    double   eb;
    uint32_t radius : 16;
    uint32_t entry[END + 1];
    uint32_t nnz;

} psz_header;

typedef struct pszmem_pool {
    size_t seg_len[END];
    size_t seg_entry[END];
    float  density{0.2};
    size_t compressed_len;
    int    nnz;

    psz_memseg __pool;
    psz_memseg data;
    psz_memseg errctrl;
    psz_memseg sp_val;
    psz_memseg sp_idx;
    psz_memseg sp_val_full;  // compat for demo purpose
    psz_memseg hf_bitstream;
    psz_memseg anchor;
    psz_memseg freq;

} pszmem_pool;

// void init(pszmem_pool*, size_t);
// void destroy(pszmem_pool*);

#ifdef __cplusplus
}
#endif

#endif /* DA377C1A_D4A3_492C_A9E1_44072067050A */
