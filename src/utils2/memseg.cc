/**
 * @file memseg.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-06-09
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include "utils2/memseg.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "utils/io.hh"

using std::string;

namespace {

using Mem = psz_memseg;

}

void pszmem__calc_len(Mem* m)
{
  m->len = m->lx * m->ly * m->lz;
  m->tsize = m->type % 10;
  m->bytes = m->tsize * m->len;
  m->sty = m->lx;
  m->stz = m->lx * m->ly;
}

void pszmem__check_len(Mem* m)
{
  // TODO psz runtime error
  if (m->len == 0) {
    pszmem__dbg(m);
    throw std::runtime_error(
        "'" + string(m->name) + "'\tLen == 0 is not allowed.");
  }
  if (m->len == 1) {
    pszmem__dbg(m);
    throw std::runtime_error(
        "'" + string(m->name) + "'\tLen == 1 is not allowed.");
  }
  if (m->lx == 1) {
    pszmem__dbg(m);
    throw std::runtime_error(
        "'" + string(m->name) + "'\tLen-x == 1 is not allowed.");
  }
}

int pszmem__ndim(Mem* m)
{
  auto ndim = 3;
  if (m->lz == 1) ndim = 2;
  if (m->ly == 1) ndim = 1;

  return ndim;
}

void pszmem__dbg(psz_memseg* m)
{
  printf("pszmem::name\t%s\n", m->name);
  printf("pszmem::{dtype, tsize}\t{%d, %d}\n", m->type, m->tsize);
  printf("pszmem::{len, bytes}\t{%lu, %lu}\n", m->len, m->bytes);
  printf("pszmem::{lx, ly, lz}\t{%u, %u, %u}\n", m->lx, m->ly, m->lz);
  printf("pszmem::{sty, stz}\t{%lu, %lu}\n", m->sty, m->stz);
  printf("pszmem::{d, h, uni}\t{%p, %p, %p}\n", m->d, m->h, m->uni);
  printf("\n");
}

Mem* pszmem_create1(psz_dtype t, uint32_t lx)
{
  auto m = new psz_memseg{.type = t, .lx = lx};
  pszmem__calc_len(m);
  pszmem__check_len(m);
  return m;
}

Mem* pszmem_create2(psz_dtype t, uint32_t lx, uint32_t ly)
{
  auto m = new psz_memseg{.type = t, .lx = lx, .ly = ly};
  pszmem__calc_len(m);
  pszmem__check_len(m);
  return m;
}

Mem* pszmem_create3(psz_dtype t, uint32_t lx, uint32_t ly, uint32_t lz)
{
  auto m = new psz_memseg{.type = t, .lx = lx, .ly = ly, .lz = lz};
  pszmem__calc_len(m);
  pszmem__check_len(m);
  return m;
}

void pszmem_set_name(psz_memseg* m, const char name[10])
{
  strcpy(m->name, name);
}

void pszmem_malloc_cuda(Mem* m)
{
  if (m->d == nullptr) {
    if (not m->isaview)
      cudaMalloc(&m->d, m->bytes);
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to malloc a view.");
  }
  else {
    throw std::runtime_error(string(m->name) + ": dptr already malloc'ed.");
  }
  cudaMemset(m->d, 0x0, m->bytes);
}

void pszmem_mallochost_cuda(Mem* m)
{
  if (m->h == nullptr) {
    if (not m->isaview)
      cudaMallocHost(&m->h, m->bytes);
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to malloc a view.");
  }
  else {
    throw std::runtime_error(string(m->name) + ": hptr already malloc'ed.");
  }
  memset(m->h, 0x0, m->bytes);
}

void pszmem_clearhost(Mem* m) { memset(m->h, 0x0, m->bytes); }

void pszmem_cleardevice(Mem* m) { cudaMemset(m->d, 0x0, m->bytes); }

void pszmem_mallocmanaged_cuda(Mem* m)
{
  if (m->uni == nullptr) {
    if (not m->isaview)
      cudaMallocManaged(&m->uni, m->bytes);
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to malloc a view.");
  }
  else {
    throw std::runtime_error(string(m->name) + ": uniptr already malloc'ed.");
  }
  cudaMemset(m->uni, 0x0, m->bytes);
}

void pszmem_free_cuda(Mem* m)
{
  if (m->d) {
    if (not m->isaview)
      cudaFree(m->d);
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to free a view.");
  }
}

void pszmem_freehost_cuda(Mem* m)
{
  if (m->h) {
    if (not m->isaview)
      cudaFreeHost(m->h);
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to free a view.");
  }
}

void pszmem_freemanaged_cuda(Mem* m)
{
  if (m->uni) {
    if (not m->isaview)
      cudaFree(m->uni);
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to free a view.");
  }
}

void pszmem_h2d_cuda(Mem* m)
{
  cudaMemcpy(m->d, m->h, m->bytes, cudaMemcpyHostToDevice);
}

void pszmem_h2d_cudaasync(Mem* m, void* stream)
{
  cudaMemcpyAsync(
      m->d, m->h, m->bytes, cudaMemcpyHostToDevice, (cudaStream_t)stream);
}

void pszmem_d2h_cuda(Mem* m)
{
  cudaMemcpy(m->h, m->d, m->bytes, cudaMemcpyDeviceToHost);
}

void pszmem_d2h_cudaasync(Mem* m, void* stream)
{
  cudaMemcpyAsync(
      m->h, m->d, m->bytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
}

void pszmem_device_deepcpy_cuda(Mem* dst, Mem* src)
{
  cudaMemcpy(dst->h, src->d, src->bytes, cudaMemcpyDeviceToDevice);
}

void pszmem_host_deepcpy_cuda(Mem* dst, Mem* src)
{
  cudaMemcpy(dst->h, src->d, src->bytes, cudaMemcpyHostToHost);
}

void pszmem_fromfile(const char* fname, Mem* m)
{
  std::ifstream ifs(fname, std::ios::binary | std::ios::in);
  if (not ifs.is_open()) {
    std::cerr << "fail to open " << fname << std::endl;
    exit(1);
  }
  ifs.read(reinterpret_cast<char*>(m->h), std::streamsize(m->bytes));
  ifs.close();
}

void pszmem_tofile(const char* fname, Mem* m)
{
  std::ofstream ofs(fname, std::ios::binary | std::ios::out);
  if (not ofs.is_open()) {
    std::cerr << "fail to open " << fname << std::endl;
    exit(1);
  }
  ofs.write(reinterpret_cast<const char*>(m->h), std::streamsize(m->bytes));
  ofs.close();
}

void pszmem_viewas(psz_memseg* body, psz_memseg* view)
{
  view->isaview = true;

  if (view->bytes >= body->bytes)
    throw std::runtime_error("The view exceeds the legal length.");

  if (body->d or body->h or body->uni) {
    view->d = body->d;
    view->h = body->h;
    view->uni = body->uni;
  }
  else {
    throw std::runtime_error("Must be malloc'ed in hptr, dptr, or uniptr.");
  }
}
