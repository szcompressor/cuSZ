/**
 * @file memseg_cxx.hip.hh
 * @author Jiannan Tian
 * @brief drop-in replacement of previous Capsule
 * @version 0.4
 * @date 2023-06-09
 * (created) 2020-11-03 (capsule.hh)
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef E9058A2D_B20F_4DA5_B818_B495C26F66CD
#define E9058A2D_B20F_4DA5_B818_B495C26F66CD

// The next-line: failsafe macro check 
#ifdef PSZ_USE_HIP

#include <hip/hip_runtime.h>

template <typename Ctype>
class pszmem_cxx {
 private:
  static const psz_dtype T = PszType<Ctype>::type;

 public:
  pszmem* m;
  int ndim;
  double maxval, minval, range;

  pszmem_cxx(
      uint32_t lx, uint32_t ly = 1, uint32_t lz = 1,
      const char name[10] = "<unamed>")
  {
    m = pszmem_create3(T, lx, ly, lz);
    pszmem_setname(m, name);
    ndim = pszmem__ndim(m);
  }

  ~pszmem_cxx()
  {
    if (not m->isaview and not m->d_borrowed) pszmem_free_hip(m);
    if (not m->isaview and not m->h_borrowed) pszmem_freehost_hip(m);
    if (not m->isaview and not m->d_borrowed and not m->h_borrowed)
      pszmem_freemanaged_hip(m);
    delete m;
  }

  pszmem_cxx* extrema_scan(double& max_value, double& min_value, double& range)
  {
    if (std::is_same<Ctype, float>::value or
        std::is_same<Ctype, double>::value) {
      // may not work for _uniptr
      Ctype result[4];
      // psz::thrustgpu::thrustgpu_get_extrema_rawptr<Ctype>((Ctype*)m->d, m->len,
      // result);
      psz::probe_extrema<HIP, Ctype>((Ctype*)m->d, m->len, result);

      min_value = result[0];
      max_value = result[1];
      range = max_value - min_value;
    }
    else {
      throw std::runtime_error(
          "`extrema_scan` only supports `float` or `double`.");
    }

    return this;
  }

  pszmem_cxx* control(
      std::vector<pszmem_control> control_stream,
      hipStream_t stream = nullptr)
  {
    for (auto& c : control_stream) {
      if (c == Malloc)
        pszmem_malloc_hip(m);
      else if (c == MallocHost)
        pszmem_mallochost_hip(m);
      else if (c == MallocManaged)
        pszmem_mallocmanaged_hip(m);
      else if (c == Free)
        pszmem_free_hip(m);
      else if (c == FreeHost)
        pszmem_freehost_hip(m);
      else if (c == FreeManaged)
        pszmem_freemanaged_hip(m);
      else if (c == ClearHost)
        pszmem_clearhost(m);
      else if (c == ClearDevice)
        pszmem_cleardevice_hip(m);
      else if (c == H2D)
        pszmem_h2d_hip(m);
      else if (c == ASYNC_H2D)
        pszmem_h2d_hipasync(m, stream);
      else if (c == D2H)
        pszmem_d2h_hip(m);
      else if (c == ASYNC_D2H)
        pszmem_d2h_hipasync(m, stream);
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

  // view
  template <typename Ctype2>
  pszmem_cxx* asaviewof(pszmem_cxx<Ctype2>* another)
  {
    m->isaview = true;
    pszmem_viewas(another->m, this->m);
    return this;
  }

  // cast
  template <typename Ctype2>
  pszmem_cxx* castto(pszmem_cxx<Ctype2>* dst, psz_space const s)
  {
    auto len1 = this->len();
    auto len2 = dst->len();

    if (len1 != len2) throw std::runtime_error("dst is not equal in length.");

    if (s == psz_space::Host) {
      if (dst->hptr() == nullptr or this->hptr() == nullptr)
        throw std::runtime_error("hptr not set.");
      for (auto i = 0; i < len1; i++) dst->hptr(i) = this->hptr(i);
    }
    else if (s == psz_space::Device) {
      throw std::runtime_error(
          "not implemented; use casting in host space instead.");
    }
    else {
      throw std::runtime_error("not a legal space");
    }
    return this;
  }

  // setter by borrowing
  pszmem_cxx* dptr(Ctype* d)
  {
    if (d == nullptr) throw std::runtime_error("`d` arg. must not be nil.");
    pszmem_borrow(this->m, d, nullptr);
    return this;
  }

  pszmem_cxx* hptr(Ctype* h)
  {
    if (h == nullptr) throw std::runtime_error("`h` arg. must not be nil.");
    pszmem_borrow(this->m, nullptr, h);
    return this;
  }

  pszmem_cxx* uniptr(Ctype* uni)
  {
    if (uni == nullptr)
      throw std::runtime_error("`uni` arg. must not be nil.");
    // to decrease the complexity of C impl.
    m->uni = uni, m->d_borrowed = m->h_borrowed = true;
    return this;
  }

  // getter
  size_t len() const { return m->len; }
  size_t bytes() const { return m->bytes; };
  Ctype* dptr() const { return (Ctype*)m->d; };
  Ctype* dbegin() const { return dptr(); };
  Ctype* dend() const { return dptr() + m->len; };
  Ctype* hptr() const { return (Ctype*)m->h; };
  Ctype* hbegin() const { return hptr(); };
  Ctype* hend() const { return hptr() + m->len; };
  Ctype* uniptr() const { return (Ctype*)m->uni; };
  Ctype* unibegin() const { return uniptr(); };
  Ctype* uniend() const { return uniptr() + m->len; };
  // getter by index
  Ctype& dptr(uint32_t i) { return ((Ctype*)m->d)[i]; };
  Ctype& dat(uint32_t i) { return ((Ctype*)m->d)[i]; };
  Ctype& hptr(uint32_t i) { return ((Ctype*)m->h)[i]; };
  Ctype& hat(uint32_t i) { return ((Ctype*)m->h)[i]; };
  Ctype& uniptr(uint32_t i) { return ((Ctype*)m->uni)[i]; };
  Ctype& uniat(uint32_t i) { return ((Ctype*)m->uni)[i]; };

  template <typename UINT3>
  UINT3 len3() const
  {
    return UINT3(m->lx, m->ly, m->lz);
  };

  template <typename UINT3>
  UINT3 st3() const
  {
    return UINT3(1, m->sty, m->stz);
  };
};

#endif

#endif /* E9058A2D_B20F_4DA5_B818_B495C26F66CD */
