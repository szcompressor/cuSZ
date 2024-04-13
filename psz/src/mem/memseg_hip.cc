/**
 * @file memseg.cc
 * @author Jiannan Tian
 * @brief As portable as possible.
 * @version 0.4
 * @date 2023-06-09
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <hip/hip_runtime.h>

#include "busyheader.hh"
#include "mem/memseg.h"

void pszmem_malloc_hip(pszmem* m)
{
  if (m->d_borrowed)
    throw std::runtime_error(
        string(m->name) + ": cannot malloc borrowed dptr.");

  if (m->d == nullptr) {
    if (not m->isaview)
      hipMalloc(&m->d, m->bytes);
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to malloc a view.");
  }
  else {
    throw std::runtime_error(string(m->name) + ": dptr already malloc'ed.");
  }
  hipMemset(m->d, 0x0, m->bytes);
}

void pszmem_mallochost_hip(pszmem* m)
{
  if (m->h_borrowed)
    throw std::runtime_error(
        string(m->name) + ": cannot malloc borrowed hptr.");

  if (m->h == nullptr) {
    if (not m->isaview)
      hipHostMalloc(&m->h, m->bytes);
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to malloc a view.");
  }
  else {
    throw std::runtime_error(string(m->name) + ": hptr already malloc'ed.");
  }
  memset(m->h, 0x0, m->bytes);
}

void pszmem_cleardevice_hip(pszmem* m) { hipMemset(m->d, 0x0, m->bytes); }

void pszmem_mallocmanaged_hip(pszmem* m)
{
  if (m->d_borrowed or m->h_borrowed)
    throw std::runtime_error(
        string(m->name) + ": cannot malloc borrowed uniptr.");

  if (m->uni == nullptr) {
    if (not m->isaview)
      hipMallocManaged(&m->uni, m->bytes);
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to malloc a view.");
  }
  else {
    throw std::runtime_error(string(m->name) + ": uniptr already malloc'ed.");
  }
  hipMemset(m->uni, 0x0, m->bytes);
}

void pszmem_free_hip(pszmem* m)
{
  if (m->d_borrowed)
    throw std::runtime_error(string(m->name) + ": cannot free borrowed dptr");

  if (m->d) {
    if (not m->isaview)
      hipFree(m->d);
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to free a view.");
  }
}

void pszmem_freehost_hip(pszmem* m)
{
  if (m->h_borrowed)
    throw std::runtime_error(string(m->name) + ": cannot free borrowed hptr.");

  if (m->h) {
    if (not m->isaview)
      hipHostFree(m->h);
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to free a view.");
  }
}

void pszmem_freemanaged_hip(pszmem* m)
{
  if (m->d_borrowed or m->h_borrowed)
    throw std::runtime_error(
        string(m->name) + ": cannot free borrowed uniptr.");

  if (m->uni) {
    if (not m->isaview)
      hipFree(m->uni);
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to free a view.");
  }
}

void pszmem_h2d_hip(pszmem* m)
{
  hipMemcpy(m->d, m->h, m->bytes, hipMemcpyHostToDevice);
}

void pszmem_h2d_hipasync(pszmem* m, void* stream)
{
  hipMemcpyAsync(
      m->d, m->h, m->bytes, hipMemcpyHostToDevice, (hipStream_t)stream);
}

void pszmem_d2h_hip(pszmem* m)
{
  hipMemcpy(m->h, m->d, m->bytes, hipMemcpyDeviceToHost);
}

void pszmem_d2h_hipasync(pszmem* m, void* stream)
{
  hipMemcpyAsync(
      m->h, m->d, m->bytes, hipMemcpyDeviceToHost, (hipStream_t)stream);
}

void pszmem_device_deepcopy_hip(pszmem* dst, pszmem* src)
{
  hipMemcpy(dst->d, src->d, src->bytes, hipMemcpyDeviceToDevice);
}

void pszmem_host_deepcopy_hip(pszmem* dst, pszmem* src)
{
  hipMemcpy(dst->h, src->h, src->bytes, hipMemcpyHostToHost);
}
