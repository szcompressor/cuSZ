#ifndef _PORTABLE_MEM_SP_INTERFACE_H
#define _PORTABLE_MEM_SP_INTERFACE_H

#include <cstdint>
#include <type_traits>

#include "c_type.h"

namespace _portable {

enum class outlier_stragegy { CONVINIENT, RIGOROUS };

template <typename T>
struct compact_GPU_DRAM;

template <typename T, typename Idx>
struct compact_GPU_DRAM2;

template <typename ValT, typename IdxT = u4, outlier_stragegy OS = outlier_stragegy::CONVINIENT>
struct compact_cell {
  // static_assert(std::is_floating_point_v<ValT>, "ValT must be f4 or f8.");

  using OutlierValT = std::conditional_t<OS == outlier_stragegy::CONVINIENT, f4, ValT>;
  OutlierValT val;
  IdxT idx;
} __attribute__((packed));

}  // namespace _portable

#endif /* _PORTABLE_MEM_SP_INTERFACE_H */
