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
  bool isaview{false}, d_borrowed{false}, h_borrowed{false};

} pszmem;

typedef uint32_t _u4;

void pszmem__calc_len(pszmem* m);
int pszmem__ndim(pszmem* m);
void pszmem__dbg(pszmem* m);
pszmem* pszmem_create1(psz_dtype t, _u4 lx);
pszmem* pszmem_create2(psz_dtype t, _u4 lx, _u4 ly);
pszmem* pszmem_create3(psz_dtype t, _u4 lx, _u4 ly, _u4 lz);
void pszmem_borrow(pszmem* m, void* _d, void* _h);
void pszmem_setname(pszmem* m, const char name[10]);
void pszmem_malloc_cuda(pszmem* m);
void pszmem_mallochost_cuda(pszmem* m);
void pszmem_mallocmanaged_cuda(pszmem* m);
void pszmem_free_cuda(pszmem* m);
void pszmem_freehost_cuda(pszmem* m);
void pszmem_freemanaged_cuda(pszmem* m);
void pszmem_clearhost(pszmem* m);
void pszmem_cleardevice(pszmem* m);
void pszmem_h2d_cuda(pszmem* m);
void pszmem_h2d_cudaasync(pszmem* m, void* stream);
void pszmem_d2h_cuda(pszmem* m);
void pszmem_d2h_cudaasync(pszmem* m, void* stream);
void pszmem_device_deepcpy_cuda(pszmem* dst, pszmem* src);
void pszmem_host_deepcpy_cuda(pszmem* dst, pszmem* src);
void pszmem_fromfile(const char* fname, pszmem* m);
void pszmem_tofile(const char* fname, pszmem* m);
void pszmem_viewas(pszmem* backend, pszmem* frontend);

// no impl. in C due to typing issue
// void pszmem_cast(pszmem* dst, pszmem* src, psz_space s);

#ifdef __cplusplus
}
#endif

#endif /* C00A17B2_29D2_47F0_B667_E6814586B4EB */
