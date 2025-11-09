/**
 * @file compact.seq.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-04-05
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef _PORTABLE_MEM_CXX_SP_CPU_H
#define _PORTABLE_MEM_CXX_SP_CPU_H

#include <cstdint>
#include <cstdlib>
#include <memory>

#include "sp_interface.h"

namespace _portable {

template <typename T, typename Idx = uint32_t>
struct compact_CPU {
 public:
  using cell = compact_cell<T, Idx>;

  std::unique_ptr<cell[]> _val_idx;
  std::unique_ptr<uint32_t[]> _num;

  const size_t reserved_len;

  compact_CPU(size_t _reserved_len) : reserved_len(_reserved_len)
  {
    _val_idx = std::make_unique<cell[]>(reserved_len + 10);
    _num = std::make_unique<uint32_t[]>(1);
    _num[0] = 0;
  };
  ~compact_CPU() {}

  // getter
  cell* val_idx() { return _val_idx.get(); }
  cell& val_idx(const size_t i) { return _val_idx[i]; }
  uint32_t& num() const { return _num[0]; }
  size_t max_allowed_num() const { return reserved_len; }
  // uint32_t get_num() { return _num[0]; }
};

}  // namespace _portable

#endif /* _PORTABLE_MEM_CXX_SP_CPU_H */
