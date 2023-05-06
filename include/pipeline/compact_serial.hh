/**
 * @file compact_serial.hh
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

#include <stdint.h>
#include <stdlib.h>

#include "compaction.hh"

template <typename T>
struct CompactSerial {
 public:
  using type = T;

  T* val;
  uint32_t* idx;
  uint32_t num{0};
  size_t reserved_len;

  // CompactSerial() {}
  // ~CompactSerial() {}

  CompactSerial& set_reserved_len(size_t _reserved_len)
  {
    reserved_len = _reserved_len;
    return *this;
  }

  CompactSerial& malloc()
  {
    val = new T[reserved_len];
    idx = new uint32_t[reserved_len];

    return *this;
  }

  CompactSerial& free()
  {
    delete[] val;
    delete[] idx;
    return *this;
  }

  // accessor
  uint32_t num_outliers() const { return num; }
};

#endif /* AF5746CE_5141_41E1_AF1D_FF476775BB6E */
