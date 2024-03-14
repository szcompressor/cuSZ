/**
 * @file l23r.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-04-04
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <cuda_runtime.h>

#include <type_traits>

// deps
#include "cusz/type.h"
#include "kernel/lrz.hh"
#include "mem/compact.hh"
#include "utils/err.hh"
#include "utils/timer.hh"
// definitions
#include "detail/l23r.cu_hip.inl"

template <typename T, typename Eq, bool TIMING, bool ZigZag>
pszerror pszcxx_predict_lorenzo(
    T* const data, dim3 const len3, f8 const eb, int const radius,
    Eq* const eq, void* _outlier, f4* time_elapsed, void* stream)
{
  static_assert(
      std::is_same<Eq, u4>::value or std::is_same<Eq, uint16_t>::value or
          std::is_same<Eq, uint8_t>::value,
      "Eq must be unsigned integer that is less than or equal to 4 bytes.");



  using Compact = typename CompactDram<PROPER_GPU_BACKEND, T>::Compact;
  using namespace psz::kernelconfig;

  auto ot = (Compact*)_outlier;

  auto d = lorenzo_utils::ndim(len3);

  // error bound
  auto ebx2 = eb * 2;
  auto ebx2_r = 1 / ebx2;
  auto leap3 = dim3(1, len3.x, len3.x * len3.y);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING((cudaStream_t)stream);

  if (d == 1) {
    psz::rolling::c_lorenzo_1d1l<
        T, Eq, T, c_lorenzo<1>::tile.x, c_lorenzo<1>::sequentiality.x>
        <<<c_lorenzo<1>::thread_grid(len3), c_lorenzo<1>::thread_block, 0,
           (cudaStream_t)stream>>>(
            data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),
            ot->num());
  }
  else if (d == 2) {
    psz::rolling::c_lorenzo_2d1l<T, Eq, T>
        <<<c_lorenzo<2>::thread_grid(len3), c_lorenzo<2>::thread_block, 0,
           (cudaStream_t)stream>>>(
            data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),
            ot->num());
  }
  else if (d == 3) {
    psz::rolling::c_lorenzo_3d1l<T, Eq, T>
        <<<c_lorenzo<3>::thread_grid(len3), c_lorenzo<3>::thread_block, 0,
           (cudaStream_t)stream>>>(
            data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),
            ot->num());
  }

  STOP_GPUEVENT_RECORDING((cudaStream_t)stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}

#define INIT(T, E, TIMING, ZIGZAG)                                   \
  template pszerror pszcxx_predict_lorenzo<T, E, TIMING, ZIGZAG>(    \
      T* const data, dim3 const len3, f8 const eb, int const radius, \
      E* const eq, void* _outlier, f4* time_elapsed, void* stream);

INIT(f4, u4, true, false)
INIT(f4, u4, true, true)
INIT(f8, u4, true, false)
INIT(f8, u4, true, true)

INIT(f4, u4, false, false)
INIT(f4, u4, false, true)
INIT(f8, u4, false, false)
INIT(f8, u4, false, true)

#undef INIT