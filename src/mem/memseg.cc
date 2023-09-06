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

#include "mem/memseg.h"
#include "busyheader.hh"

#include <fstream>

void pszmem__calc_len(pszmem* m)
{
  m->len = m->lx * m->ly * m->lz;
  m->tsize = m->type % 10;
  m->bytes = m->tsize * m->len;
  m->sty = m->lx;
  m->stz = m->lx * m->ly;
}

void pszmem__check_len(pszmem* m)
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

int pszmem__ndim(pszmem* m)
{
  auto ndim = 3;
  if (m->lz == 1) ndim = 2;
  if (m->ly == 1) ndim = 1;

  return ndim;
}

void pszmem__dbg(pszmem* m)
{
  printf("pszmem::name\t%s\n", m->name);
  printf("pszmem::{dtype, tsize}\t{%d, %d}\n", m->type, m->tsize);
  printf("pszmem::{len, bytes}\t{%lu, %lu}\n", m->len, m->bytes);
  printf("pszmem::{lx, ly, lz}\t{%u, %u, %u}\n", m->lx, m->ly, m->lz);
  printf("pszmem::{sty, stz}\t{%lu, %lu}\n", m->sty, m->stz);
  printf("pszmem::{d, h, uni}\t{%p, %p, %p}\n", m->d, m->h, m->uni);
  printf("\n");
}

pszmem* pszmem_create1(psz_dtype t, u4 lx)
{
  auto m = new pszmem{.type = t, .lx = lx};
  pszmem__calc_len(m);
  pszmem__check_len(m);
  return m;
}

pszmem* pszmem_create2(psz_dtype t, u4 lx, u4 ly)
{
  auto m = new pszmem{.type = t, .lx = lx, .ly = ly};
  pszmem__calc_len(m);
  pszmem__check_len(m);
  return m;
}

pszmem* pszmem_create3(psz_dtype t, u4 lx, u4 ly, u4 lz)
{
  auto m = new pszmem{.type = t, .lx = lx, .ly = ly, .lz = lz};
  pszmem__calc_len(m);
  pszmem__check_len(m);
  return m;
}

void pszmem_borrow(pszmem* m, void* src_d, void* src_h)
{
  if (src_d) m->d = src_d, m->d_borrowed = true;
  if (src_h) m->d = src_h, m->h_borrowed = true;
}

void pszmem_setname(pszmem* m, const char name[10]) { strcpy(m->name, name); }

void pszmem_clearhost(pszmem* m) { memset(m->h, 0x0, m->bytes); }

void pszmem_fromfile(const char* fname, pszmem* m)
{
  std::ifstream ifs(fname, std::ios::binary | std::ios::in);
  if (not ifs.is_open()) {
    std::cerr << "fail to open " << fname << std::endl;
    exit(1);
  }
  ifs.read(reinterpret_cast<char*>(m->h), std::streamsize(m->bytes));
  ifs.close();
}

void pszmem_tofile(const char* fname, pszmem* m)
{
  std::ofstream ofs(fname, std::ios::binary | std::ios::out);
  if (not ofs.is_open()) {
    std::cerr << "fail to open " << fname << std::endl;
    exit(1);
  }
  ofs.write(reinterpret_cast<const char*>(m->h), std::streamsize(m->bytes));
  ofs.close();
}

void pszmem_viewas(pszmem* body, pszmem* view)
{
  view->isaview = true;

  if (view->bytes > body->bytes)
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
