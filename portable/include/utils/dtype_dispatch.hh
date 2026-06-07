#ifndef _PORTABLE_UTILS_DTYPE_DISPATCH_HH
#define _PORTABLE_UTILS_DTYPE_DISPATCH_HH

// No __CUDACC__ guard — this header is usable from NVCC-compiled files
// as host-only code. std::function is safe in the host path.

#include <functional>
#include <type_traits>
#include <vector>

#include "c_type.h"

namespace _ptb::utils {

// Lightweight type tag — C++17 compatible substitute for std::type_identity.
template <typename T>
struct type_tag {
  using type = T;
};

// Dispatch on _ptb_dtype at runtime, invoking the registered handler.
//
// Usage:
//   dtype_dispatch()
//     .on<float,  F4>([&](type_tag<float>)  { ... float path ...  })
//     .on<double, F8>([&](type_tag<double>) { ... double path ... })
//     .call(dtype);
//
// T in .on<T, E> is passed as a type_tag to the lambda — use it to
// derive the concrete type inside the handler without duplicating
// the enum->type mapping at every call site.

class dtype_dispatch {
  using fn_t = std::function<void()>;

  std::vector<std::pair<_ptb_dtype, fn_t>> handlers_;

 public:
  // Register a handler for dtype enum value E.
  // T must be a floating-point type matching E (e.g. float <-> F4).
  // The lambda receives type_tag<T>{} as its sole argument.
  template <typename T, _ptb_dtype E, typename Fn>
  dtype_dispatch&& on(Fn&& fn)
  {
    static_assert(std::is_floating_point_v<T>, "T must be float or double");
    handlers_.push_back({E, [fn = std::forward<Fn>(fn)]() { fn(type_tag<T>{}); }});
    return std::move(*this);
  }

  // Invoke the handler registered for dtype.
  // Returns true if a matching handler was found, false otherwise.
  bool call(_ptb_dtype dtype) const
  {
    for (auto& [e, fn] : handlers_)
      if (e == dtype) {
        fn();
        return true;
      }
    return false;
  }
};

}  // namespace _ptb::utils

#endif  // _PORTABLE_UTILS_DTYPE_DISPATCH_HH
