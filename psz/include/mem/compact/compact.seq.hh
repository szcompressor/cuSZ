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

#ifndef AF5746CE_5141_41E1_AF1D_FF476775BB6E
#define AF5746CE_5141_41E1_AF1D_FF476775BB6E

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "cusz/type.h"
#include "mem/definition.hh"

template <typename T>
struct CompactSerial {
 public:
  using type = T;
  using control_stream_t = std::vector<pszmem_control>;

  T* _val;
  uint32_t* _idx;
  uint32_t _num{0};
  size_t reserved_len;

  CompactSerial(size_t _reserved_len) : reserved_len(_reserved_len){};
  ~CompactSerial() { free(); }

  CompactSerial& malloc()
  {
    _val = new T[reserved_len];
    _idx = new uint32_t[reserved_len];

    return *this;
  }

  CompactSerial& free()
  {
    delete[] _val;
    delete[] _idx;
    return *this;
  }

  CompactSerial& control(
      control_stream_t controls, void* placeholder = nullptr)
  {
    for (auto& c : controls) {
      if (c == Malloc)
        malloc();
      else if (c == MallocHost) {
      }
      else if (c == Free)
        free();
      else if (c == FreeHost) {
      }
      else if (c == D2H) {
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

#endif /* AF5746CE_5141_41E1_AF1D_FF476775BB6E */
