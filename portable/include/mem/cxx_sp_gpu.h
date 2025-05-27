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
  // GPU_unique_uptr<uint32_t[]> u_num;  // use unified memory
  GPU_unique_dptr<uint32_t[]> d_num;  // use unified memory
  GPU_unique_hptr<uint32_t[]> h_num;  // use unified memory

  const size_t reserved_len;

  compact_GPU_DRAM(size_t _reserved_len, bool need_host_alloc = false) :
      reserved_len(_reserved_len)
  {
    d_val = MAKE_UNIQUE_DEVICE(T, reserved_len + 10);
    d_idx = MAKE_UNIQUE_DEVICE(uint32_t, reserved_len + 10);
    // u_num = MAKE_UNIQUE_UNIFIED(uint32_t, 1);
    // *(u_num.get()) = 0;
    d_num = MAKE_UNIQUE_DEVICE(uint32_t, 1);
    h_num = MAKE_UNIQUE_HOST(uint32_t, 1);

    if (need_host_alloc) {
      h_val = MAKE_UNIQUE_HOST(T, reserved_len + 10);
      h_idx = MAKE_UNIQUE_HOST(uint32_t, reserved_len + 10);
    }
  }

  ~compact_GPU_DRAM() {}

 public:
  // accessor
  uint32_t num_outliers() const
  {
    memcpy_allkinds<D2H>(h_num.get(), d_num.get(), 1);
    return *(h_num.get());
  }
  T* val() const { return d_val.get(); }
  uint32_t* idx() const { return d_idx.get(); }
  uint32_t* num() const { return d_num.get(); }
  size_t max_allowed_num() const { return reserved_len; }
};

}  // namespace _portable

#endif /* _PORTABLE_MEM_CXX_SP_GPU_H */
