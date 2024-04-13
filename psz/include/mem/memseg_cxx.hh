/**
 * @file memseg_cxx_cu.hh
 * @author Jiannan Tian
 * @brief drop-in replacement of previous Capsule
 * @version 0.4
 * @date 2023-06-09
 * (created) 2020-11-03 (capsule.hh)
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef B0F9D525_8AEF_4F79_A668_6C36D73DDF9F
#define B0F9D525_8AEF_4F79_A668_6C36D73DDF9F

#include <cstddef>

#include "busyheader.hh"
#include "cusz/type.h"
#include "memseg_cxx/definition.hh"
#include "memseg.h"
#include "stat/compare.hh"
#include "typing.hh"

#if defined(PSZ_USE_CUDA)
#include "memseg_cxx/memseg_cxx.cu.hh"
#elif defined(PSZ_USE_HIP)
#include "memseg_cxx/memseg_cxx.hip.hh"
#elif defined(PSZ_USE_1API)
#include "memseg_cxx/memseg_cxx.dp.hh"
#endif

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

#endif /* B0F9D525_8AEF_4F79_A668_6C36D73DDF9F */
