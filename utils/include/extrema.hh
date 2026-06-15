#ifndef _PSZ_UTILS_EXTREMA_HH
#define _PSZ_UTILS_EXTREMA_HH

#include <cstddef>
#include <tuple>

namespace psz::cuda {

template <typename T>
struct GPU_get_extrema {
  static auto kernel(T* in, size_t len, void* stream = nullptr) -> std::tuple<T, T, T, T>;
};

}  // namespace psz::cuda

#endif /* _PSZ_UTILS_EXTREMA_HH */
