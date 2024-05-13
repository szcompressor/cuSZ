/**
 * @file memseg_cxx.cu.hh
 * @author Jiannan Tian
 * @brief drop-in replacement of previous Capsule
 * @version 0.4
 * @date 2023-06-09
 * (created) 2020-11-03 (capsule.hh)
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef EC1B3A67_146B_48BF_A336_221E9D38C41F
#define EC1B3A67_146B_48BF_A336_221E9D38C41F

// The next-line: failsafe macro check
#include <linux/limits.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "busyheader.hh"
#include "cusz/type.h"
#include "mem/definition.hh"
#include "stat/compare.hh"
#include "typing.hh"

template <typename Ctype = uint8_t>
class pszmem_cxx {
 private:
  struct impl;
  std::unique_ptr<impl> pimpl;

  const psz_dtype type{PszType<Ctype>::type};

 public:
  double maxval, minval, range;

 public:
  pszmem_cxx(u4 _lx, const char _name[32] = "<unnamed>");
  pszmem_cxx(u4 _lx, u4 _ly, const char _name[32] = "<unnamed>");
  pszmem_cxx(u4 _lx, u4 _ly, u4 _lz, const char _name[32] = "<unnamed>");
  ~pszmem_cxx();

  pszmem_cxx* extrema_scan(
      double& max_value, double& min_value, double& range);

  pszmem_cxx* control(
      std::vector<pszmem_control> control_stream, void* stream = nullptr);

  pszmem_cxx* file(const char* fname, pszmem_control control);

  // setter (borrowing)
  pszmem_cxx* dptr(Ctype* d);
  pszmem_cxx* hptr(Ctype* h);
  pszmem_cxx* uniptr(Ctype* uni);
  // setter
  pszmem_cxx* set_len(size_t ext_len);

  // getter
  size_t len() const;
  size_t bytes() const;
  Ctype* dptr() const;
  Ctype* dbegin() const;
  Ctype* dend() const;
  Ctype* hptr() const;
  Ctype* hbegin() const;
  Ctype* hend() const;
  Ctype* uniptr() const;
  Ctype* unibegin() const;
  Ctype* uniend() const;
  // getter by index
  Ctype& dptr(uint32_t i);
  Ctype& dat(uint32_t i);
  Ctype& hptr(uint32_t i);
  Ctype& hat(uint32_t i);
  Ctype& uniptr(uint32_t i);
  Ctype& uniat(uint32_t i);

  template <typename UINT3>
  UINT3 len3() const;

  template <typename UINT3>
  UINT3 st3() const;
};

typedef pszmem_cxx<u1> MemU1;
typedef pszmem_cxx<u2> MemU2;
typedef pszmem_cxx<u4> MemU4;
typedef pszmem_cxx<u8> MemU8;
typedef pszmem_cxx<ull> MemUll;
typedef pszmem_cxx<i1> MemI1;
typedef pszmem_cxx<i2> MemI2;
typedef pszmem_cxx<i4> MemI4;
typedef pszmem_cxx<i8> MemI8;
typedef pszmem_cxx<szt> MemSzt;
typedef pszmem_cxx<szt> MemZu;
typedef pszmem_cxx<f4> MemF4;
typedef pszmem_cxx<f8> MemF8;

#endif /* EC1B3A67_146B_48BF_A336_221E9D38C41F */
