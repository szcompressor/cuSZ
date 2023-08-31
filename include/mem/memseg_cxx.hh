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

#include "busyheader.hh"
#include "cusz/type.h"
#include "memseg.h"
#include "stat/compare.hh"
#include "typing.hh"

enum pszmem_control_stream {
  Malloc,
  MallocHost,
  MallocManaged,
  Free,
  FreeHost,
  FreeManaged,
  ClearHost,
  ClearDevice,
  H2D,
  D2H,
  ASYNC_H2D,
  ASYNC_D2H,
  ToFile,
  FromFile,
  ExtremaScan,
};

using pszmem_control = pszmem_control_stream;

#if defined(PSZ_USE_CUDA)
  #include "memseg_cxx_cu.hh"
#elif defined(PSZ_USE_HIP)
  #include "memseg_cxx_hip.hh"
#elif defined(PSZ_USE_1API)
  // #include "memseg_cxx_1api.hh"
#endif


#endif /* B0F9D525_8AEF_4F79_A668_6C36D73DDF9F */
