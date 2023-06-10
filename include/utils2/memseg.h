/**
 * @file capsule.h
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-06-09
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef C00A17B2_29D2_47F0_B667_E6814586B4EB
#define C00A17B2_29D2_47F0_B667_E6814586B4EB

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "../cusz/type.h"

typedef struct psz_memory_segment {
  char name[10];
  psz_dtype type;
  int tsize;
  void* buf;  // compat for wip
  void *d, *h, *uni;
  size_t len{1}, bytes{1};
  uint32_t lx{1}, ly{1}, lz{1};
  size_t sty{1}, stz{1};  // stride

} psz_memseg;

typedef psz_memseg pszmem;

void pszmem__calc_len(psz_memseg* m);
int pszmem__ndim(psz_memseg* m);
void pszmem__dbg(psz_memseg* m);
psz_memseg* pszmem_create1(psz_dtype t, uint32_t lx);
psz_memseg* pszmem_create2(psz_dtype t, uint32_t lx, uint32_t ly);
psz_memseg* pszmem_create3(psz_dtype t, uint32_t lx, uint32_t ly, uint32_t lz);
void pszmem_set_name(psz_memseg* m, const char name[10]);
void pszmem_malloc_cuda(psz_memseg* m);
void pszmem_mallochost_cuda(psz_memseg* m);
void pszmem_mallocmanaged_cuda(psz_memseg* m);
void pszmem_free_cuda(psz_memseg* m);
void pszmem_freehost_cuda(psz_memseg* m);
void pszmem_freemanaged_cuda(psz_memseg* m);
void pszmem_h2d_cuda(psz_memseg* m);
void pszmem_h2d_cudaasync(psz_memseg* m, void* stream);
void pszmem_d2h_cuda(psz_memseg* m);
void pszmem_d2h_cudaasync(psz_memseg* m, void* stream);
void pszmem_device_deepcpy_cuda(psz_memseg* dst, psz_memseg* src);
void pszmem_host_deepcpy_cuda(psz_memseg* dst, psz_memseg* src);
void pszmem_fromfile(const char* fname, psz_memseg* m);
void pszmem_tofile(const char* fname, psz_memseg* m);

#ifdef __cplusplus
}
#endif

#endif /* C00A17B2_29D2_47F0_B667_E6814586B4EB */
