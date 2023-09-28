#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "busyheader.hh"
#include "mem/memseg.h"
#include "utils/err.hh"

void pszmem_malloc_1api(pszmem* m)
{
  dpct::device_ext& dev_ct1 = dpct::get_current_device();
  sycl::queue& q_ct1 = dev_ct1.default_queue();
  if (m->d_borrowed)
    throw std::runtime_error(
        string(m->name) + ": cannot malloc borrowed dptr.");

  if (m->d == nullptr) {
    if (not m->isaview)
      m->d = (void*)sycl::malloc_device(m->bytes, q_ct1);
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to malloc a view.");
  }
  else {
    throw std::runtime_error(string(m->name) + ": dptr already malloc'ed.");
  }
  q_ct1.memset(m->d, 0x0, m->bytes).wait_and_throw();
}

void pszmem_mallochost_1api(pszmem* m)
{
  if (m->h_borrowed)
    throw std::runtime_error(
        string(m->name) + ": cannot malloc borrowed hptr.");

  if (m->h == nullptr) {
    if (not m->isaview)
      m->h = (void*)sycl::malloc_host(m->bytes, dpct::get_default_queue());
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to malloc a view.");
  }
  else {
    throw std::runtime_error(string(m->name) + ": hptr already malloc'ed.");
  }
  memset(m->h, 0x0, m->bytes);
}

void pszmem_cleardevice_1api(pszmem* m)
{
  dpct::get_default_queue().memset(m->d, 0x0, m->bytes).wait_and_throw();
}

void pszmem_mallocshared_1api(pszmem* m)
{
  dpct::device_ext& dev_ct1 = dpct::get_current_device();
  sycl::queue& q_ct1 = dev_ct1.default_queue();
  if (m->d_borrowed or m->h_borrowed)
    throw std::runtime_error(
        string(m->name) + ": cannot malloc borrowed uniptr.");

  if (m->uni == nullptr) {
    if (not m->isaview)
      m->uni = (void*)sycl::malloc_shared(m->bytes, q_ct1);
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to malloc a view.");
  }
  else {
    throw std::runtime_error(string(m->name) + ": uniptr already malloc'ed.");
  }
  q_ct1.memset(m->uni, 0x0, m->bytes).wait_and_throw();
}

void pszmem_free_1api(pszmem* m)
{
  if (m->d_borrowed)
    throw std::runtime_error(string(m->name) + ": cannot free borrowed dptr");

  if (m->d) {
    if (not m->isaview)
      sycl::free(m->d, dpct::get_default_queue());
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to free a view.");
  }
}

void pszmem_freehost_1api(pszmem* m)
{
  if (m->h_borrowed)
    throw std::runtime_error(string(m->name) + ": cannot free borrowed hptr.");

  if (m->h) {
    if (not m->isaview)
      sycl::free(m->h, dpct::get_default_queue());
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to free a view.");
  }
}

void pszmem_freeshared_1api(pszmem* m)
{
  if (m->d_borrowed or m->h_borrowed)
    throw std::runtime_error(
        string(m->name) + ": cannot free borrowed uniptr.");

  if (m->uni) {
    if (not m->isaview)
      sycl::free(m->uni, dpct::get_default_queue());
    else
      throw std::runtime_error(
          string(m->name) + ": forbidden to free a view.");
  }
}

void pszmem_h2d_1api(pszmem* m)
{
  dpct::get_default_queue()  //
      .memcpy(m->d, m->h, m->bytes)
      .wait_and_throw();
}

// [psz::TODO] wait or wait_and_throw?
void pszmem_h2d_1apiasync(pszmem* m, void* stream)
{
  ((dpct::queue_ptr)stream)  //
      ->memcpy(m->d, m->h, m->bytes)
      .wait_and_throw();
}

void pszmem_d2h_1api(pszmem* m)
{
  dpct::get_default_queue()  //
      .memcpy(m->h, m->d, m->bytes)
      .wait_and_throw();
}

void pszmem_d2h_1apiasync(pszmem* m, void* stream)
{
  ((dpct::queue_ptr)stream)  //
      ->memcpy(m->h, m->d, m->bytes)
      .wait_and_throw();
}

void pszmem_device_deepcopy_1api(pszmem* dst, pszmem* src)
{
  dpct::get_default_queue()
      .memcpy(dst->d, src->d, src->bytes)
      .wait_and_throw();
}

void pszmem_host_deepcopy_1api(pszmem* dst, pszmem* src)
{
  dpct::get_default_queue()
      .memcpy(dst->h, src->h, src->bytes)
      .wait_and_throw();
}
