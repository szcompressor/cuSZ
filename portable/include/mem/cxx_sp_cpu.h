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
#include <vector>

#include "c_type.h"
#include "cusz/type.h"

namespace _portable {

template <typename T>
struct compact_seq {
 public:
  using type = T;
  using control_stream_t = std::vector<_portable_mem_control>;

  T* _val;
  uint32_t* _idx;
  uint32_t _num{0};
  size_t reserved_len;

  compact_seq(size_t _reserved_len) : reserved_len(_reserved_len){};
  ~compact_seq() { free(); }

  compact_seq& malloc()
  {
    _val = new T[reserved_len];
    _idx = new uint32_t[reserved_len];

    return *this;
  }

  compact_seq& free()
  {
    delete[] _val;
    delete[] _idx;
    return *this;
  }

  compact_seq& control(control_stream_t controls, void* placeholder = nullptr)
  {
    for (auto& c : controls) {
      if (c == Malloc) { malloc(); }
      else if (c == MallocHost) {
      }
      else if (c == Free) {
        free();
      }
      else if (c == FreeHost) {
      }
    }

    return *this;
  }

  // getter
  uint32_t num_outliers() const { return _num; }
  T* val() { return _val; }
  T& val(szt i) { return _val[i]; }
  uint32_t* idx() { return _idx; }
  uint32_t& idx(szt i) { return _idx[i]; }

  uint32_t& num() { return _num; }
};

}  // namespace _portable

#endif /* _PORTABLE_MEM_CXX_SP_CPU_H */
