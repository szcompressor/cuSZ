#ifndef _PORTABLE_UTILS_DEMANGLE_HH
#define _PORTABLE_UTILS_DEMANGLE_HH

#ifdef __CUDACC__
#error "demangle.hh must not be included from CUDA translation units"
#endif

#include <cstdlib>
#include <string>

// __cxa_demangle is a GCC/Clang extension; absent on MSVC and some embedded
// toolchains.  __has_include lets us probe at compile time and fall back to
// returning the mangled name as-is on platforms that lack cxxabi.h.
#if __has_include(<cxxabi.h>)
#include <cxxabi.h>
#define _PORTABLE_HAS_CXXABI 1
#endif

namespace _portable::utils {

inline std::string demangle(const char* name)
{
#if defined(_PORTABLE_HAS_CXXABI)
  int         status = -4;
  char*       res    = abi::__cxa_demangle(name, nullptr, nullptr, &status);
  std::string ret(status == 0 ? res : name);
  free(res);  // free(nullptr) is safe when status != 0
  return ret;
#else
  return name;
#endif
}

}  // namespace _portable::utils

#endif  // _PORTABLE_UTILS_DEMANGLE_HH
