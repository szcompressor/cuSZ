#ifndef _PORTABLE_DETAIL_STR2NUM_HH
#define _PORTABLE_DETAIL_STR2NUM_HH

#ifdef __CUDACC__
#error "detail/str2num.hh must not be included from CUDA translation units"
#endif

#include <cerrno>
#include <cstdlib>
#include <optional>

#include "c_type.h"

namespace _portable::detail {

inline std::optional<i8> str_to_int(const char* s)
{
  if (not s or not *s) return std::nullopt;
  char* end;
  errno       = 0;
  long long v = std::strtoll(s, &end, 10);
  if (*end or errno) return std::nullopt;
  return static_cast<i8>(v);
}

inline std::optional<f8> str_to_num(const char* s)
{
  if (not s or not *s) return std::nullopt;
  char* end;
  errno    = 0;
  double v = std::strtod(s, &end);
  if (*end or errno) return std::nullopt;
  return v;
}

}  // namespace _portable::detail

#endif  // _PORTABLE_DETAIL_STR2NUM_HH
