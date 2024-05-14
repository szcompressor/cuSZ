#include <cuda_runtime.h>

#include "experimental/mem_multibackend.hh"
#include "mem/array_cxx.h"
#include "mem/definition.hh"
#include "mem/memobj.hh"

// The next-line: failsafe macro check
#include <linux/limits.h>

#include <fstream>
#include <iostream>

#include "busyheader.hh"
#include "cusz/type.h"
#include "stat/compare.hh"
#include "typing.hh"

namespace portable {

template <typename Ctype>
struct memobj<Ctype>::impl {
  char name[32];
  const psz_dtype type{PszType<Ctype>::type};
  Ctype *d{nullptr}, *h{nullptr}, *uni{nullptr};
  size_t _len{1}, _bytes{1};
  uint32_t lx{1}, ly{1}, lz{1};
  size_t sty{1}, stz{1};  // stride
  bool d_borrowed{false}, h_borrowed{false};

  int ndim;
  double maxval, minval, range;

#if defined(PSZ_USE_CUDA)
  static const psz_platform BACKEND = CUDA;
#elif defined(PSZ_USE_HIP)
  static const psz_platform BACKEND = HIP;
#elif defined(PSZ_USE_1API)
  static const psz_platform BACKEND = ONEAPI;
#endif

  void _constructor(u4 _lx, u4 _ly, u4 _lz, const char _name[32] = "<unnamed>")
  {
    this->lx = _lx;
    this->ly = _ly;
    this->lz = _lz;
    strcpy(name, _name);
    _calc_len(), _check_len();
  }

  void _calc_len()
  {
    _len = lx * ly * lz;
    _bytes = sizeof(Ctype) * _len;
    sty = lx;
    stz = lx * ly;

    ndim = 3;
    if (lz == 1) ndim = 2;
    if (ly == 1) ndim = 1;
  }

  void _dbg()
  {
    printf("memobj::name\t%s\n", name);
    printf("memobj::{dtype}\t{%d}\n", type);
    printf("memobj::{len, bytes}\t{%lu, %lu}\n", _len, _bytes);
    printf("memobj::{lx, ly, lz}\t{%u, %u, %u}\n", lx, ly, lz);
    printf("memobj::{sty, stz}\t{%lu, %lu}\n", sty, stz);
    printf("memobj::{d, h}\t{%p, %p, %p}\n", d, h, uni);
    printf("\n");
  }

  void _check_len()
  {
    if (_len == 0) {
      _dbg();
      throw std::runtime_error(
          "'" + string(name) + "'\tLen == 0 is not allowed.");
    }
  }

  void malloc_device(void* stream = nullptr)
  {
    if (d_borrowed)
      throw std::runtime_error(
          string(name) + ": cannot malloc borrowed dptr.");

    if (d == nullptr) {
      // cudaMalloc(&d, _bytes);
      d = ::malloc_device<Ctype>(_len, stream);
    }
    else
      throw std::runtime_error(string(name) + ": dptr already malloc'ed.");

    cudaMemset(d, 0x0, _bytes);
  }

  void malloc_host(void* stream = nullptr)
  {
    if (h_borrowed)
      throw std::runtime_error(
          string(name) + ": cannot malloc borrowed hptr.");

    if (h == nullptr) {
      // cudaMallocHost(&h, _bytes);
      h = ::malloc_host<Ctype>(_len, stream);
    }
    else
      throw std::runtime_error(string(name) + ": hptr already malloc'ed.");

    memset(h, 0x0, _bytes);
  }

  void malloc_shared(void* stream = nullptr)
  {
    if (d_borrowed or h_borrowed)
      throw std::runtime_error(
          string(name) + ": cannot malloc borrowed uniptr.");

    if (uni == nullptr) {
      // cudaMallocManaged(&uni, _bytes);
      uni = ::malloc_shared<Ctype>(_len, stream);
    }
    else
      throw std::runtime_error(string(name) + ": uniptr already malloc'ed.");

    cudaMemset(uni, 0x0, _bytes);
  }

  // clang-format off
  void free_device(void* stream = nullptr) { if (d and not d_borrowed) ::free_device(d, stream); }
  void free_host(void* stream = nullptr)   { if (h and not h_borrowed) ::free_host(h, stream); }
  void free_shared(void* stream = nullptr) { if (uni and not d_borrowed and not h_borrowed) ::free_shared(uni, stream); }
  void h2d() { _memcpy_allkinds<Ctype, H2D>(d, h, _len); }
  void d2h() { _memcpy_allkinds<Ctype, D2H>(h, d, _len); }
  void h2d_async(void* stream) { _memcpy_allkinds_async<Ctype, H2D>(d, h, _len, stream); }
  void d2h_async(void* stream) { _memcpy_allkinds_async<Ctype, D2H>(h, d, _len, stream); }
  void clear_host()   { memset(h, 0x0, _bytes); }
  void clear_device() { cudaMemset(d, 0x0, _bytes); }
  void clear_shared() { cudaMemset(uni, 0x0, _bytes); }
  // clang-format on

  void _borrow(Ctype* src_d, Ctype* src_h)
  {
    if (src_d) d = src_d, d_borrowed = true;
    if (src_h) d = src_h, h_borrowed = true;
  }

  void fromfile(const char* fname)
  {
    std::ifstream ifs(fname, std::ios::binary | std::ios::in);
    if (not ifs.is_open()) {
      std::cerr << "fail to open " << fname << std::endl;
      exit(1);
    }
    ifs.read(reinterpret_cast<char*>(h), std::streamsize(_bytes));
    ifs.close();
  }

  void tofile(const char* fname)
  {
    std::ofstream ofs(fname, std::ios::binary | std::ios::out);
    if (not ofs.is_open()) {
      std::cerr << "fail to open " << fname << std::endl;
      exit(1);
    }
    ofs.write(reinterpret_cast<const char*>(h), std::streamsize(_bytes));
    ofs.close();
  }

  void extrema_scan(double& max_value, double& min_value, double& range)
  {
    if (std::is_same<Ctype, float>::value or
        std::is_same<Ctype, double>::value) {
      // may not work for _uniptr
      Ctype result[4];
      // psz::thrustgpu::thrustgpu_get_extrema_rawptr<Ctype>((Ctype*)m->d,
      // m->len, result);
      psz::probe_extrema<CUDA, Ctype>(d, _len, result);

      min_value = result[0];
      max_value = result[1];
      range = max_value - min_value;
    }
    else {
      throw std::runtime_error(
          "`extrema_scan` only supports `float` or `double`.");
    }
  }

  // setter by borrowing
  void dptr(Ctype* d)
  {
    if (d == nullptr) throw std::runtime_error("`d` arg. must not be nil.");
    _borrow(d, nullptr);
  }

  void hptr(Ctype* h)
  {
    if (h == nullptr) throw std::runtime_error("`h` arg. must not be nil.");
    _borrow(nullptr, h);
  }

  void uniptr(Ctype* uni)
  {
    if (uni == nullptr)
      throw std::runtime_error("`uni` arg. must not be nil.");
    // to decrease the complexity of C impl.
    uni = uni, d_borrowed = h_borrowed = true;
  }

  void set_len(size_t ext_len)
  {
    _len = ext_len, _bytes = sizeof(Ctype) * _len;
  }

  // getter
  size_t len() const { return _len; }
  size_t bytes() const { return _bytes; };
  Ctype* dptr() const { return d; };
  Ctype* dbegin() const { return dptr(); };
  Ctype* dend() const { return dptr() + _len; };
  Ctype* hptr() const { return h; };
  Ctype* hbegin() const { return hptr(); };
  Ctype* hend() const { return hptr() + _len; };
  Ctype* uniptr() const { return uni; };
  Ctype* unibegin() const { return uniptr(); };
  Ctype* uniend() const { return uniptr() + _len; };

  // getter of interop
  // TODO ctor from array3/array1
  // clang-format off
  ::portable::array3<Ctype> array3_h() const { return {hptr(), {lx, ly, lz}}; };
  ::portable::array3<Ctype> array3_d() const { return {dptr(), {lx, ly, lz}}; };
  ::portable::array3<Ctype> array3_uni() const { return {uniptr(), {lx, ly, lz}}; };
  ::portable::array1<Ctype> array1_h() const { return {hptr(), _len}; };
  ::portable::array1<Ctype> array1_d() const { return {dptr(), _len}; };
  ::portable::array1<Ctype> array1_uni() const { return {uniptr(), _len}; };
  // clang-format on

  // getter by index
  Ctype& dptr(uint32_t i) { return d[i]; };
  Ctype& dat(uint32_t i) { return d[i]; };
  Ctype& hptr(uint32_t i) { return h[i]; };
  Ctype& hat(uint32_t i) { return h[i]; };
  Ctype& uniptr(uint32_t i) { return uni[i]; };
  Ctype& uniat(uint32_t i) { return uni[i]; };

  template <typename UINT3>
  UINT3 len3() const
  {
    return UINT3{lx, ly, lz};
  };

  template <typename UINT3>
  UINT3 st3() const
  {
    return UINT3(1, sty, stz);
  };
};

//////////////////////////////// back to main class
///////////////////////////////

template <typename Ctype>
memobj<Ctype>::memobj(u4 _lx, const char _name[32]) :
    pimpl{std::make_unique<impl>()}
{
  pimpl->_constructor(_lx, 1, 1, _name);
}

template <typename Ctype>
memobj<Ctype>::memobj(u4 _lx, u4 _ly, const char _name[32]) :
    pimpl{std::make_unique<impl>()}
{
  pimpl->_constructor(_lx, _ly, 1, _name);
}

template <typename Ctype>
memobj<Ctype>::memobj(u4 _lx, u4 _ly, u4 _lz, const char _name[32]) :
    pimpl{std::make_unique<impl>()}
{
  pimpl->_constructor(_lx, _ly, _lz, _name);
}

template <typename Ctype>
memobj<Ctype>::~memobj()
{
  pimpl->free_device();
  pimpl->free_host();
  pimpl->free_shared();
}

template <typename Ctype>
memobj<Ctype>* memobj<Ctype>::extrema_scan(
    double& max_value, double& min_value, double& range)
{
  pimpl->extrema_scan(max_value, min_value, range);

  return this;
}

template <typename Ctype>
memobj<Ctype>* memobj<Ctype>::control(
    std::vector<pszmem_control> control_stream, void* stream)
{
  for (auto& c : control_stream) {
    if (c == Malloc)
      pimpl->malloc_device();
    else if (c == MallocHost)
      pimpl->malloc_host();
    else if (c == MallocManaged)
      pimpl->malloc_shared();
    else if (c == Free)
      pimpl->free_device();
    else if (c == FreeHost)
      pimpl->free_host();
    else if (c == FreeManaged)
      pimpl->free_shared();
    else if (c == ClearHost)
      pimpl->clear_host();
    else if (c == ClearDevice)
      pimpl->clear_device();
    else if (c == H2D)
      pimpl->h2d();
    else if (c == ASYNC_H2D)
      pimpl->h2d_async(stream);
    else if (c == D2H)
      pimpl->d2h();
    else if (c == ASYNC_D2H)
      pimpl->d2h_async(stream);
    else if (c == ExtremaScan)
      pimpl->extrema_scan(maxval, minval, range);
    else if (c == DBG)
      pimpl->_dbg();
  }
  return this;
}

template <typename Ctype>
memobj<Ctype>* memobj<Ctype>::file(const char* fname, pszmem_control control)
{
  if (control == ToFile)
    pimpl->tofile(fname);
  else if (control == FromFile)
    pimpl->fromfile(fname);
  else
    throw std::runtime_error("must be `FromFile` or `ToFile`");

  return this;
}

template <typename Ctype>
memobj<Ctype>* memobj<Ctype>::dptr(Ctype* d)
{
  pimpl->dptr(d);
  return this;
}

template <typename Ctype>
memobj<Ctype>* memobj<Ctype>::hptr(Ctype* h)
{
  pimpl->hptr(h);
  return this;
}

template <typename Ctype>
memobj<Ctype>* memobj<Ctype>::uniptr(Ctype* uni)
{
  pimpl->uniptr(uni);
  return this;
}

template <typename Ctype>
memobj<Ctype>* memobj<Ctype>::set_len(size_t ext_len)
{
  pimpl->set_len(ext_len);
  return this;
}

template <typename Ctype>
size_t memobj<Ctype>::len() const
{
  return pimpl->len();
}
template <typename Ctype>
size_t memobj<Ctype>::bytes() const
{
  return pimpl->bytes();
};
template <typename Ctype>
Ctype* memobj<Ctype>::dptr() const
{
  return pimpl->dptr();
};
template <typename Ctype>
Ctype* memobj<Ctype>::dbegin() const
{
  return pimpl->dbegin();
};
template <typename Ctype>
Ctype* memobj<Ctype>::dend() const
{
  return pimpl->dend();
};
template <typename Ctype>
Ctype* memobj<Ctype>::hptr() const
{
  return pimpl->hptr();
};
template <typename Ctype>
Ctype* memobj<Ctype>::hbegin() const
{
  return pimpl->hbegin();
};
template <typename Ctype>
Ctype* memobj<Ctype>::hend() const
{
  return pimpl->hend();
};
template <typename Ctype>
Ctype* memobj<Ctype>::uniptr() const
{
  return pimpl->uniptr();
};
template <typename Ctype>
Ctype* memobj<Ctype>::unibegin() const
{
  return pimpl->unibegin();
};
template <typename Ctype>
Ctype* memobj<Ctype>::uniend() const
{
  return pimpl->uniend();
};
template <typename Ctype>
Ctype& memobj<Ctype>::dptr(uint32_t i)
{
  return pimpl->d[i];
};
template <typename Ctype>
Ctype& memobj<Ctype>::dat(uint32_t i)
{
  return pimpl->d[i];
};
template <typename Ctype>
Ctype& memobj<Ctype>::hptr(uint32_t i)
{
  return pimpl->h[i];
};
template <typename Ctype>
Ctype& memobj<Ctype>::hat(uint32_t i)
{
  return pimpl->h[i];
};
template <typename Ctype>
Ctype& memobj<Ctype>::uniptr(uint32_t i)
{
  return pimpl->uni[i];
};
template <typename Ctype>
Ctype& memobj<Ctype>::uniat(uint32_t i)
{
  return pimpl->uni[i];
};

template <typename Ctype>
template <typename UINT3>
UINT3 memobj<Ctype>::len3() const
{
  return pimpl->template len3<UINT3>();
};

template <typename Ctype>
template <typename UINT3>
UINT3 memobj<Ctype>::st3() const
{
  return pimpl->template st3<UINT3>();
};

template <typename Ctype>
::portable::array3<Ctype> memobj<Ctype>::array3_h() const
{
  return pimpl->array3_h();
};

template <typename Ctype>
::portable::array3<Ctype> memobj<Ctype>::array3_d() const
{
  return pimpl->array3_d();
};
template <typename Ctype>
::portable::array3<Ctype> memobj<Ctype>::array3_uni() const
{
  return pimpl->array3_uni();
};
template <typename Ctype>
::portable::array1<Ctype> memobj<Ctype>::array1_h() const
{
  return pimpl->array1_h();
};
template <typename Ctype>
::portable::array1<Ctype> memobj<Ctype>::array1_d() const
{
  return pimpl->array1_d();
};
template <typename Ctype>
::portable::array1<Ctype> memobj<Ctype>::array1_uni() const
{
  return pimpl->array1_uni();
};

}  // namespace portable