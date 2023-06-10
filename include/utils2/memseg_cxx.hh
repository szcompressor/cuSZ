/**
 * @file memseg_cxx.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-06-09
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef EC1B3A67_146B_48BF_A336_221E9D38C41F
#define EC1B3A67_146B_48BF_A336_221E9D38C41F

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

#include "memseg.h"
#include "stat/compare_gpu.hh"
#include "type_traits.hh"

enum pszmem_control_stream {
  Malloc,
  MallocHost,
  MallocManaged,
  Free,
  FreeHost,
  FreeManaged,
  H2D,
  D2H,
  ASYNC_H2D,
  ASYNC_D2H,
  ToFile,
  FromFile,
  ExtremaScan,
};

using pszmem_control = pszmem_control_stream;

template <typename Ctype>
class pszmem_cxx {
 private:
  static const psz_dtype T = PszType<Ctype>::type;

 public:
  pszmem* m;
  int ndim;
  double maxval, minval, range;

  pszmem_cxx(
      uint32_t lx, uint32_t ly = 1, uint32_t lz = 1, const char name[10] = "")
  {
    m = pszmem_create3(T, lx, ly, lz);
    pszmem_set_name(m, name);
    ndim = pszmem__ndim(m);
  }

  ~pszmem_cxx() { delete m; }

  pszmem_cxx* extrema_scan(double& max_value, double& min_value, double& range)
  {
    // may not work for _uniptr
    Ctype result[4];
    psz::thrustgpu_get_extrema_rawptr<Ctype>((Ctype*)m->d, m->len, result);

    min_value = result[0];
    max_value = result[1];
    range = max_value - min_value;

    return this;
  }

  pszmem_cxx* control(
      std::vector<pszmem_control> control_stream,
      cudaStream_t stream = nullptr)
  {
    for (auto& c : control_stream) {
      if (c == Malloc)
        pszmem_malloc_cuda(m);
      else if (c == MallocHost)
        pszmem_mallochost_cuda(m);
      else if (c == MallocManaged)
        pszmem_mallocmanaged_cuda(m);
      else if (c == Free)
        pszmem_free_cuda(m);
      else if (c == FreeHost)
        pszmem_freehost_cuda(m);
      else if (c == FreeManaged)
        pszmem_freemanaged_cuda(m);
      else if (c == H2D)
        pszmem_h2d_cuda(m);
      else if (c == ASYNC_H2D)
        pszmem_h2d_cudaasync(m, stream);
      else if (c == D2H)
        pszmem_d2h_cuda(m);
      else if (c == ASYNC_D2H)
        pszmem_d2h_cudaasync(m, stream);
      else if (c == ExtremaScan)
        extrema_scan(maxval, minval, range);
    }
    return this;
  }

  pszmem_cxx* file(const char* fname, pszmem_control control)
  {
    if (control == ToFile)
      pszmem_tofile(fname, m);
    else if (control == FromFile)
      pszmem_fromfile(fname, m);
    else
      throw std::runtime_error("must be `FromFile` or `ToFile`");

    return this;
  }

  pszmem_cxx* debug()
  {
    pszmem__dbg(m);
    return this;
  }

 public:  // getters
  size_t len() { return m->len; }
  Ctype* dptr() { return (Ctype*)m->d; };
  Ctype* hptr() { return (Ctype*)m->h; };
};

// template <psz_space space>
// psz_mem

#endif /* EC1B3A67_146B_48BF_A336_221E9D38C41F */
