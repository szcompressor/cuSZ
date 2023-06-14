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

typedef enum psz_runtime_mem_segment { HEADER = 0, ERRCTRL = 1, SP_VAL = 2, SP_IDX = 3, END = 4 } psz_mem_segment;

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

typedef struct psz_runtime_mem_layout {
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

} psz_mem_layout;

void init(psz_mem_layout*, size_t);
void destroy(psz_mem_layout*);

void psz_memseg_assign(psz_memseg* m, psz_dtype const t, void* b, uint32_t const l);
void psz_malloc_cuda(psz_memseg* m);
void psz_mallochost_cuda(psz_memseg* m);
void psz_mallocmanaged_cuda(psz_memseg* m);
void psz_free_cuda(psz_memseg* m);
void psz_freehost_cuda(psz_memseg* m);

#ifdef __cplusplus
}
#endif

#endif /* DA377C1A_D4A3_492C_A9E1_44072067050A */