/**
 * @file compact.cu.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2022-12-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef _PORTABLE_MEM_CXX_SP_GPU_H
#define _PORTABLE_MEM_CXX_SP_GPU_H

#include <iostream>
#include <stdexcept>
#include <vector>

#include "cusz/type.h"
#include "exception/exception.hh"
#include "mem/cxx_array.h"
#include "mem/cxx_backends.h"

namespace _portable {

template <typename T>
struct compact_GPU_DRAM;

template <typename T>
using compact_gpu = compact_GPU_DRAM<T>;

template <typename T>
struct compact_GPU_DRAM {
 public:
  using control_stream_t = std::vector<_portable_mem_control>;

  // `h_` for host-accessible
  T* d_val;
  uint32_t* d_idx;
  uint32_t* d_num;

  T* h_val;
  uint32_t* h_idx;
  uint32_t* h_num;

  size_t reserved_len;

  compact_GPU_DRAM(size_t _reserved_len) : reserved_len(_reserved_len) {}
  compact_GPU_DRAM(_portable::compact_array1<T> ca)
  {
    d_val = ca.val, d_idx = ca.idx, d_num = ca.num;
    reserved_len = ca.reserved_len;
  };

  // ~compact_GPU_DRAM() {}

  compact_GPU_DRAM& malloc()
  {
    d_val = malloc_device<T>(reserved_len);
    d_idx = malloc_device<uint32_t>(reserved_len);
    d_num = malloc_device<uint32_t>(1);

    return *this;
  }

  compact_GPU_DRAM& mallochost()
  {
    h_val = malloc_host<T>(reserved_len);
    h_idx = malloc_host<uint32_t>(reserved_len);
    h_num = malloc_host<uint32_t>(1);
    *h_num = 0;

    return *this;
  }

  compact_GPU_DRAM& free()
  {
    free_device(d_idx), free_device(d_val), free_device(d_num);
    return *this;
  }

  compact_GPU_DRAM& freehost()
  {
    free_host(h_idx), free_host(h_val), free_host(h_num);
    return *this;
  }

 public:
  // memcpy
  pszerror make_host_accessible(void* stream = nullptr)
  try {
    memcpy_allkinds_async<D2H>(h_num, d_num, 1, stream);
    sync_by_stream(stream);

    if (*h_num > reserved_len) throw psz::exception_too_many_outliers();

    return CUSZ_SUCCESS;
  }
  NONEXIT_CATCH(psz::exception_too_many_outliers, CUSZ_OUTLIER_TOO_MANY)

  compact_GPU_DRAM& control(control_stream_t controls, void* stream = nullptr)
  {
    for (auto& c : controls) {
      if (c == Malloc)
        malloc();
      else if (c == MallocHost)
        mallochost();
      else if (c == Free)
        free();
      else if (c == FreeHost)
        freehost();
      else if (c == D2H)
        make_host_accessible(stream);
    }

    return *this;
  }

  // accessor
  uint32_t num_outliers() { return *h_num; }
  T* val() { return d_val; }
  uint32_t* idx() { return d_idx; }
  uint32_t* num() { return d_num; }
};

}  // namespace _portable

#endif /* _PORTABLE_MEM_CXX_SP_GPU_H */
