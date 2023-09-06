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

#include <cuda_runtime.h>

#include "busyheader.hh"
#include "mem/memseg.h"
#include "utils/err.hh"

void pszmem_malloc_cuda(pszmem* m)
{
  if (m->d_borrowed)
    throw std::runtime_error(
        string(m->name) + ": cannot malloc borrowed dptr.");

  if (m->d == nullptr) {
    if (not m->isaview)
      CHECK_GPU(cudaMalloc(&m->d, m->bytes));
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to malloc a view.");
  }
  else {
    throw std::runtime_error(string(m->name) + ": dptr already malloc'ed.");
  }
  CHECK_GPU(cudaMemset(m->d, 0x0, m->bytes));
}

void pszmem_mallochost_cuda(pszmem* m)
{
  if (m->h_borrowed)
    throw std::runtime_error(
        string(m->name) + ": cannot malloc borrowed hptr.");

  if (m->h == nullptr) {
    if (not m->isaview)
      CHECK_GPU(cudaMallocHost(&m->h, m->bytes));
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to malloc a view.");
  }
  else {
    throw std::runtime_error(string(m->name) + ": hptr already malloc'ed.");
  }
  memset(m->h, 0x0, m->bytes);
}

void pszmem_cleardevice_cuda(pszmem* m)
{
  CHECK_GPU(cudaMemset(m->d, 0x0, m->bytes));
}

void pszmem_mallocmanaged_cuda(pszmem* m)
{
  if (m->d_borrowed or m->h_borrowed)
    throw std::runtime_error(
        string(m->name) + ": cannot malloc borrowed uniptr.");

  if (m->uni == nullptr) {
    if (not m->isaview)
      CHECK_GPU(cudaMallocManaged(&m->uni, m->bytes));
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to malloc a view.");
  }
  else {
    throw std::runtime_error(string(m->name) + ": uniptr already malloc'ed.");
  }
  CHECK_GPU(cudaMemset(m->uni, 0x0, m->bytes));
}

void pszmem_free_cuda(pszmem* m)
{
  if (m->d_borrowed)
    throw std::runtime_error(string(m->name) + ": cannot free borrowed dptr");

  if (m->d) {
    if (not m->isaview)
      CHECK_GPU(cudaFree(m->d));
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to free a view.");
  }
}

void pszmem_freehost_cuda(pszmem* m)
{
  if (m->h_borrowed)
    throw std::runtime_error(string(m->name) + ": cannot free borrowed hptr.");

  if (m->h) {
    if (not m->isaview)
      CHECK_GPU(cudaFreeHost(m->h));
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to free a view.");
  }
}

void pszmem_freemanaged_cuda(pszmem* m)
{
  if (m->d_borrowed or m->h_borrowed)
    throw std::runtime_error(
        string(m->name) + ": cannot free borrowed uniptr.");

  if (m->uni) {
    if (not m->isaview)
      CHECK_GPU(cudaFree(m->uni));
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to free a view.");
  }
}

void pszmem_h2d_cuda(pszmem* m)
{
  CHECK_GPU(cudaMemcpy(m->d, m->h, m->bytes, cudaMemcpyHostToDevice));
}

void pszmem_h2d_cudaasync(pszmem* m, void* stream)
{
  CHECK_GPU(cudaMemcpyAsync(
      m->d, m->h, m->bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream));
}

void pszmem_d2h_cuda(pszmem* m)
{
  CHECK_GPU(cudaMemcpy(m->h, m->d, m->bytes, cudaMemcpyDeviceToHost));
}

void pszmem_d2h_cudaasync(pszmem* m, void* stream)
{
  CHECK_GPU(cudaMemcpyAsync(
      m->h, m->d, m->bytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream));
}

void pszmem_device_deepcopy_cuda(pszmem* dst, pszmem* src)
{
  CHECK_GPU(cudaMemcpy(dst->d, src->d, src->bytes, cudaMemcpyDeviceToDevice));
}

void pszmem_host_deepcopy_cuda(pszmem* dst, pszmem* src)
{
  CHECK_GPU(cudaMemcpy(dst->h, src->h, src->bytes, cudaMemcpyHostToHost));
}
