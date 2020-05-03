#ifndef BASE2_HH
#define BASE2_HH
#include <cstdlib>

namespace prototype {
auto __div64by = [](double& __n, uint64_t const& __p) { reinterpret_cast<uint64_t&>(__n) -= (__p << 52u); };
auto __div32by = [](float& __n, uint32_t const& __p) { reinterpret_cast<uint32_t&>(__n) -= (__p << 23u); };
}  // namespace prototype

#endif
