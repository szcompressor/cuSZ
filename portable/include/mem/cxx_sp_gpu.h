#ifndef _PORTABLE_MEM_CXX_SP_GPU_H
#define _PORTABLE_MEM_CXX_SP_GPU_H

#include <vector>

#include "../c_type.h"
#include "mem/cxx_backends.h"
#include "mem/plan.h"
#include "mem/sp_interface.h"

namespace _ptb {

template <typename T, typename Idx = uint32_t>
struct compact_GPU_DRAM2 : public buf_base {
 public:
  using cell = compact_cell<T, Idx>;

 private:
  enum : int { VAL_IDX, NUM };
  static constexpr size_t tile1d_size = 1024;
  static constexpr size_t padding     = tile1d_size;

  GPU_unique_dptr<u1[]>       d_data, d_state;
  GPU_unique_hptr<cell[]>     h_val_idx;
  GPU_unique_hptr<uint32_t[]> h_num;

  const size_t reserved_len_wanted;
  const size_t reserved_len_actual;

  static tables frames(size_t reserved_len_actual)
  { return {{{VAL_IDX, U1, reserved_len_actual * sizeof(cell)}}, {{NUM, U4, 1}}}; }

 public:
  compact_GPU_DRAM2(size_t reserved_len, bool need_host_alloc = false, bool standalone = true) :
      buf_base(frames(reserved_len + padding)),
      reserved_len_wanted(reserved_len),
      reserved_len_actual(reserved_len + padding)
  {
    if (standalone) init();
    h_num = MAKE_UNIQUE_HOST(uint32_t, 1);

    if (need_host_alloc) h_val_idx = MAKE_UNIQUE_HOST(cell, reserved_len_actual);
  }

  ~compact_GPU_DRAM2() override = default;

  void init() override
  {
    d_data  = MAKE_UNIQUE_DEVICE(u1, data().bytes());
    d_state = MAKE_UNIQUE_DEVICE(u1, state().bytes());
    attach(d_data.get(), d_state.get());
  }

  void reset(void*) override {}

 public:
  void reset_num(void* stream = nullptr) { memset_device_async(num_d(), 1, 0, stream); }

 public:
  // accessor
  uint32_t host_get_num() const
  {
    memcpy_allkinds<D2H>(h_num.get(), num_d(), 1);
    return *(h_num.get());
  }
  cell* val_idx_d() const { return data().ptr_of<cell>(VAL_IDX); }
  cell* val_idx_h() const { return h_val_idx.get(); }
  uint32_t* num_d() const { return state().ptr_of<uint32_t>(NUM); }
  uint32_t* num_h() const { return h_num.get(); }
  uint32_t num_h(size_t i) const { return h_num[i]; }
  size_t max_allowed_num() const { return reserved_len_wanted; }

  static mem_plan::total bound(size_t reserved_len)
  { return mem_plan(frames(reserved_len + padding)); }
  static size_t val_idx_bytes(size_t reserved_len) { return bound(reserved_len).data; }
};

}  // namespace _ptb

#endif /* _PORTABLE_MEM_CXX_SP_GPU_H */
