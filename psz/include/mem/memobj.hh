/**
 * @file memobj.hh.hh
 * @author Jiannan Tian
 * @brief drop-in replacement of previous Capsule
 * @version 0.4
 * @date 2024-05-12
 * (created) 2020-11-03 (capsule.hh)
 */

#ifndef EC1B3A67_146B_48BF_A336_221E9D38C41F
#define EC1B3A67_146B_48BF_A336_221E9D38C41F

// The next-line: failsafe macro check
#include <linux/limits.h>

#include <memory>
#include <vector>

#include "cusz/type.h"
#include "mem/array_cxx.h"
#include "mem/definition.hh"
#include "typing.hh"

namespace portable {

template <typename Ctype = uint8_t>
class memobj {
 private:
  struct impl;
  std::unique_ptr<impl> pimpl;
  using control_stream_t = std::vector<pszmem_control>;
  using control_t = pszmem_control;

  const psz_dtype type{PszType<Ctype>::type};

 public:
  double maxval, minval, range;

 public:
 // for {1,2,3}-D
  memobj(u4, const char[32] = "<unnamed>", control_stream_t = {});
  memobj(u4, u4, const char[32] = "<unnamed>", control_stream_t = {});
  memobj(u4, u4, u4, const char[32] = "<unnamed>", control_stream_t = {});
  ~memobj();

  memobj* extrema_scan(double& maxval, double& minval, double& range);
  memobj* control(control_stream_t, void* stream = nullptr);
  memobj* file(const char* fname, control_t control);

  // setter (borrowing)
  memobj* dptr(Ctype* d);
  memobj* hptr(Ctype* h);
  memobj* uniptr(Ctype* uni);
  // setter
  memobj* set_len(size_t ext_len);

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

  // getter of interop
  ::portable::array3<Ctype> array3_h() const;
  ::portable::array3<Ctype> array3_d() const;
  ::portable::array3<Ctype> array3_uni() const;
  ::portable::array1<Ctype> array1_h() const;
  ::portable::array1<Ctype> array1_d() const;
  ::portable::array1<Ctype> array1_uni() const;

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

/** @deprecated */
typedef memobj<u1> MemU1;
typedef memobj<u2> MemU2;
typedef memobj<u4> MemU4;
typedef memobj<u8> MemU8;
typedef memobj<ull> MemUll;
typedef memobj<i1> MemI1;
typedef memobj<i2> MemI2;
typedef memobj<i4> MemI4;
typedef memobj<i8> MemI8;
typedef memobj<szt> MemSzt;
typedef memobj<szt> MemZu;
typedef memobj<f4> MemF4;
typedef memobj<f8> MemF8;

typedef memobj<u1> mem_u1;
typedef memobj<u2> mem_u2;
typedef memobj<u4> mem_u4;
typedef memobj<u8> mem_u8;
typedef memobj<ull> mem_ull;
typedef memobj<i1> mem_i1;
typedef memobj<i2> mem_i2;
typedef memobj<i4> mem_i4;
typedef memobj<i8> mem_i8;
typedef memobj<szt> mem_szt;
typedef memobj<szt> mem_zu;
typedef memobj<f4> mem_f4;
typedef memobj<f8> mem_f8;

}  // namespace portable

#define pszmem_cxx memobj

#endif /* EC1B3A67_146B_48BF_A336_221E9D38C41F */
