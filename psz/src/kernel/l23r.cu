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

#define L23R_LAUNCH_KERNEL                                                 \
  if (d == 1) {                                                            \
    psz::rolling::c_lorenzo_1d1l<                                          \
        T, E, T, c_lorenzo<1>::tile.x, c_lorenzo<1>::sequentiality.x>      \
        <<<c_lorenzo<1>::thread_grid(len3), c_lorenzo<1>::thread_block, 0, \
           (cudaStream_t)stream>>>(                                        \
            data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),   \
            ot->num());                                                    \
  }                                                                        \
  else if (d == 2) {                                                       \
    psz::rolling::c_lorenzo_2d1l<T, E, T>                                  \
        <<<c_lorenzo<2>::thread_grid(len3), c_lorenzo<2>::thread_block, 0, \
           (cudaStream_t)stream>>>(                                        \
            data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),   \
            ot->num());                                                    \
  }                                                                        \
  else if (d == 3) {                                                       \
    psz::rolling::c_lorenzo_3d1l<T, E, T>                                  \
        <<<c_lorenzo<3>::thread_grid(len3), c_lorenzo<3>::thread_block, 0, \
           (cudaStream_t)stream>>>(                                        \
            data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),   \
            ot->num());                                                    \
  }

template <typename T, typename E, psz_timing_mode TIMING, bool ZigZag>
pszerror pszcxx_predict_lorenzo__internal(
    T* const data, dim3 const len3, f8 const eb, int const radius, E* const eq,
    void* _outlier, f4* time_elapsed, void* stream)
{
  using Compact = typename CompactDram<PROPER_GPU_BACKEND, T>::Compact;
  using namespace psz::kernelconfig;

  auto ot = (Compact*)_outlier;
  auto d = lorenzo_utils::ndim(len3);

  // error bound
  auto ebx2 = eb * 2, ebx2_r = 1 / ebx2;
  auto leap3 = dim3(1, len3.x, len3.x * len3.y);

  if constexpr (TIMING == CPU_BARRIER_AND_TIMING) {
    CREATE_GPUEVENT_PAIR;
    START_GPUEVENT_RECORDING((cudaStream_t)stream);

    L23R_LAUNCH_KERNEL;

    STOP_GPUEVENT_RECORDING((cudaStream_t)stream);
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
    TIME_ELAPSED_GPUEVENT(time_elapsed);
    DESTROY_GPUEVENT_PAIR;
  }
  else if constexpr (TIMING == CPU_BARRIER) {
    L23R_LAUNCH_KERNEL;
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  }
  else if constexpr (TIMING == GPU_AUTOMONY) {
    L23R_LAUNCH_KERNEL;
  }
  else {
    throw std::runtime_error(
        "[2403] fail on purpose; show now run into this branch.");
  }

  return CUSZ_SUCCESS;
}

#define L23R_INIT(T, E, TIMING, ZIGZAG)                                     \
  template pszerror pszcxx_predict_lorenzo__internal<T, E, TIMING, ZIGZAG>( \
      T* const data, dim3 const len3, f8 const eb, int const radius,        \
      E* const eq, void* _outlier, f4* time_elapsed, void* stream);

L23R_INIT(f4, u2, CPU_BARRIER_AND_TIMING, false)
L23R_INIT(f4, u2, CPU_BARRIER_AND_TIMING, true)
L23R_INIT(f8, u2, CPU_BARRIER_AND_TIMING, false)
L23R_INIT(f8, u2, CPU_BARRIER_AND_TIMING, true)

L23R_INIT(f4, u2, CPU_BARRIER, false)
L23R_INIT(f4, u2, CPU_BARRIER, true)
L23R_INIT(f8, u2, CPU_BARRIER, false)
L23R_INIT(f8, u2, CPU_BARRIER, true)

L23R_INIT(f4, u2, GPU_AUTOMONY, false)
L23R_INIT(f4, u2, GPU_AUTOMONY, true)
L23R_INIT(f8, u2, GPU_AUTOMONY, false)
L23R_INIT(f8, u2, GPU_AUTOMONY, true)

L23R_INIT(f4, u4, CPU_BARRIER_AND_TIMING, false)
L23R_INIT(f4, u4, CPU_BARRIER_AND_TIMING, true)
L23R_INIT(f8, u4, CPU_BARRIER_AND_TIMING, false)
L23R_INIT(f8, u4, CPU_BARRIER_AND_TIMING, true)

L23R_INIT(f4, u4, CPU_BARRIER, false)
L23R_INIT(f4, u4, CPU_BARRIER, true)
L23R_INIT(f8, u4, CPU_BARRIER, false)
L23R_INIT(f8, u4, CPU_BARRIER, true)

L23R_INIT(f4, u4, GPU_AUTOMONY, false)
L23R_INIT(f4, u4, GPU_AUTOMONY, true)
L23R_INIT(f8, u4, GPU_AUTOMONY, false)
L23R_INIT(f8, u4, GPU_AUTOMONY, true)

#undef L23R_INIT
#undef L23R_LAUNCH_KERNEL