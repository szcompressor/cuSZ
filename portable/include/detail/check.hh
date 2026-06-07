#ifndef _PORTABLE_DETAIL_CHECK_HH
#define _PORTABLE_DETAIL_CHECK_HH

#ifdef __CUDACC__
#error "detail/check.hh must not be included from CUDA translation units"
#endif

#include <initializer_list>
#include <stdexcept>

namespace _ptb::detail {

// Returns true if val equals any element in legal.
// If throw_fail is true and val is not found, throws std::runtime_error(error_msg).
template <typename T>
inline bool check_in(
    const T& val, std::initializer_list<T> legal, const char* error_msg = nullptr,
    bool throw_fail = false)
{
  for (const auto& v : legal)
    if (val == v) return true;
  if (throw_fail and error_msg) throw std::runtime_error(error_msg);
  return false;
}

}  // namespace _ptb::detail

#endif  // _PORTABLE_DETAIL_CHECK_HH
