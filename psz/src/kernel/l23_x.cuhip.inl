/**
 * @file l23.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-01
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <cuda_runtime.h>

#include "cusz/type.h"
#include "detail/l23_x.cuhip.inl"
#include "kernel/lrz.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

#define L23X_LAUNCH_KERNEL                                                 \
  if (d == 1) {                                                            \
    psz::KERNEL_CUHIP_x_lorenzo_1d1l<                                      \
        T, E, T, x_lorenzo<1>::tile.x, x_lorenzo<1>::sequentiality.x>      \
        <<<x_lorenzo<1>::thread_grid(len3), x_lorenzo<1>::thread_block, 0, \
           (cudaStream_t)stream>>>(                                        \
            eq, outlier, len3, leap3, radius, ebx2, xdata);                \
  }                                                                        \
  else if (d == 2) {                                                       \
    psz::KERNEL_CUHIP_x_lorenzo_2d1l<T, E, T>                              \
        <<<x_lorenzo<2>::thread_grid(len3), x_lorenzo<2>::thread_block, 0, \
           (cudaStream_t)stream>>>(                                        \
            eq, outlier, len3, leap3, radius, ebx2, xdata);                \
  }                                                                        \
  else if (d == 3) {                                                       \
    psz::KERNEL_CUHIP_x_lorenzo_3d1l<T, E, T>                              \
        <<<x_lorenzo<3>::thread_grid(len3), x_lorenzo<3>::thread_block, 0, \
           (cudaStream_t)stream>>>(                                        \
            eq, outlier, len3, leap3, radius, ebx2, xdata);                \
  }

namespace psz::cuhip {

template <typename T, typename E, psz_timing_mode TIMING>
pszerror GPU_x_lorenzo_nd(
    E* eq, dim3 const len3, T* outlier, f8 const eb, int const radius,
    T* xdata, f4* time_elapsed, void* stream)
{
  using namespace psz::kernelconfig;

  // error bound
  auto ebx2 = eb * 2, ebx2_r = 1 / ebx2;
  auto leap3 = dim3(1, len3.x, len3.x * len3.y);
  auto d = lorenzo_utils::ndim(len3);

  if constexpr (TIMING == SYNC_BY_STREAM) {
    CREATE_GPUEVENT_PAIR;
    START_GPUEVENT_RECORDING((cudaStream_t)stream);

    L23X_LAUNCH_KERNEL;

    STOP_GPUEVENT_RECORDING((cudaStream_t)stream);
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
    TIME_ELAPSED_GPUEVENT(time_elapsed);
    DESTROY_GPUEVENT_PAIR;
  }
  else if constexpr (TIMING == CPU_BARRIER) {
    L23X_LAUNCH_KERNEL;
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  }
  else if constexpr (TIMING == GPU_AUTOMONY) {
    L23X_LAUNCH_KERNEL;
  }
  else {
    throw std::runtime_error(
        "[2403] fail on purpose; show now run into this branch.");
  }

  return CUSZ_SUCCESS;
}

}  // namespace psz::cuhip

#define INSTANTIATE_GPU_L23X_3params(T, E, TIMING)                        \
  template pszerror psz::cuhip::GPU_x_lorenzo_nd<T, E, TIMING>(           \
      E * eq, dim3 const len3, T* outlier, f8 const eb, int const radius, \
      T* xdata, f4* time_elapsed, void* stream);

#define INSTANTIATE_GPU_L23X_2params(T, E)            \
  INSTANTIATE_GPU_L23X_3params(T, E, SYNC_BY_STREAM); \
  INSTANTIATE_GPU_L23X_3params(T, E, CPU_BARRIER);    \
  INSTANTIATE_GPU_L23X_3params(T, E, GPU_AUTOMONY);

#define INSTANTIATE_GPU_L23X_1param(T) \
  INSTANTIATE_GPU_L23X_2params(T, u2); \
  INSTANTIATE_GPU_L23X_2params(T, u4);

#undef L23X_LAUNCH_KERNEL
