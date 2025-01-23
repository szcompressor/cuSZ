#ifndef _PORTABLE_MEM_CXX_SP_GPU_H
#define _PORTABLE_MEM_CXX_SP_GPU_H

#include <iostream>
#include <vector>

#include "cusz/type.h"
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

  GPU_unique_dptr<T[]> d_val;
  GPU_unique_hptr<T[]> h_val;
  GPU_unique_dptr<uint32_t[]> d_idx;
  GPU_unique_hptr<uint32_t[]> h_idx;
  GPU_unique_uptr<uint32_t[]> u_num;  // use unified memory

  size_t reserved_len;

  compact_GPU_DRAM(size_t _reserved_len, bool need_host_alloc = false) :
      reserved_len(_reserved_len)
  {
    d_val = MAKE_UNIQUE_DEVICE(T, reserved_len);
    d_idx = MAKE_UNIQUE_DEVICE(uint32_t, reserved_len);
    u_num = MAKE_UNIQUE_UNIFIED(uint32_t, 1);
    *(u_num.get()) = 0;

    if (need_host_alloc) {
      h_val = MAKE_UNIQUE_HOST(T, reserved_len);
      h_idx = MAKE_UNIQUE_HOST(uint32_t, reserved_len);
    }
  }

  ~compact_GPU_DRAM() {}

 public:
  // accessor
  uint32_t num_outliers() { return *(u_num.get()); }
  T* val() { return d_val.get(); }
  uint32_t* idx() { return d_idx.get(); }
  uint32_t* num() { return u_num.get(); }
};

}  // namespace _portable

#endif /* _PORTABLE_MEM_CXX_SP_GPU_H */
