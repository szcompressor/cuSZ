#ifndef _PORTABLE_MEM_CXX_SP_GPU_H
#define _PORTABLE_MEM_CXX_SP_GPU_H

#include <vector>

#include "../c_type.h"
#include "mem/cxx_backends.h"
#include "mem/sp_interface.h"

namespace _portable {

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
  GPU_unique_dptr<uint32_t[]> d_num;
  GPU_unique_hptr<uint32_t[]> h_num;

  const size_t reserved_len;

  compact_GPU_DRAM(size_t _reserved_len, bool need_host_alloc = false) :
      reserved_len(_reserved_len)
  {
    d_val = MAKE_UNIQUE_DEVICE(T, reserved_len + 10);
    d_idx = MAKE_UNIQUE_DEVICE(uint32_t, reserved_len + 10);
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

template <typename T, typename Idx = uint32_t>
struct compact_GPU_DRAM2 {
 public:
  using cell = compact_cell<T, Idx>;

 private:
  static constexpr size_t tile1d_size = 1024;
  static constexpr size_t padding = tile1d_size;

  GPU_unique_dptr<cell[]> d_val_idx;
  GPU_unique_hptr<cell[]> h_val_idx;
  GPU_unique_dptr<uint32_t[]> d_num;
  GPU_unique_hptr<uint32_t[]> h_num;

  const size_t reserved_len_wanted;
  const size_t reserved_len_actual;

 public:
  compact_GPU_DRAM2(size_t reserved_len, bool need_host_alloc = false) :
      reserved_len_wanted(reserved_len), reserved_len_actual(reserved_len + padding)
  {
    d_val_idx = MAKE_UNIQUE_DEVICE(cell, reserved_len_actual);
    d_num = MAKE_UNIQUE_DEVICE(uint32_t, 1);
    h_num = MAKE_UNIQUE_HOST(uint32_t, 1);

    if (need_host_alloc) h_val_idx = MAKE_UNIQUE_HOST(cell, reserved_len_actual);
  }

  ~compact_GPU_DRAM2() {}

 public:
  // accessor
  uint32_t host_get_num() const
  {
    memcpy_allkinds<D2H>(h_num.get(), d_num.get(), 1);
    return *(h_num.get());
  }
  cell* val_idx_d() const { return d_val_idx.get(); }
  cell* val_idx_h() const { return h_val_idx.get(); }
  uint32_t* num_d() const { return d_num.get(); }
  uint32_t* num_h() const { return h_num.get(); }
  uint32_t num_h(size_t i) const { return h_num[i]; }
  size_t max_allowed_num() const { return reserved_len_wanted; }
};

}  // namespace _portable

#endif /* _PORTABLE_MEM_CXX_SP_GPU_H */
