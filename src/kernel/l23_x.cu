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
#include "detail/l23_x.cu_hip.inl"
#include "kernel/lrz.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

template <typename T, typename Eq, bool TIMING>
pszerror pszcxx_reverse_predict_lorenzo(
    Eq* eq, dim3 const len3, T* outlier, f8 const eb, int const radius,
    T* xdata, f4* time_elapsed, void* stream)
{
  using namespace psz::kernelconfig;

  // error bound
  auto ebx2 = eb * 2, ebx2_r = 1 / ebx2;
  auto leap3 = dim3(1, len3.x, len3.x * len3.y);

  auto d = lorenzo_utils::ndim(len3);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING((cudaStream_t)stream);

  if (d == 1) {
    psz::cuda_hip::__kernel::x_lorenzo_1d1l<
        T, Eq, T, x_lorenzo<1>::tile.x, x_lorenzo<1>::sequentiality.x>
        <<<x_lorenzo<1>::thread_grid(len3), x_lorenzo<1>::thread_block, 0,
           (cudaStream_t)stream>>>(
            eq, outlier, len3, leap3, radius, ebx2, xdata);
  }
  else if (d == 2) {
    psz::cuda_hip::__kernel::x_lorenzo_2d1l<T, Eq, T>
        <<<x_lorenzo<2>::thread_grid(len3), x_lorenzo<2>::thread_block, 0,
           (cudaStream_t)stream>>>(
            eq, outlier, len3, leap3, radius, ebx2, xdata);
  }
  else if (d == 3) {
    psz::cuda_hip::__kernel::x_lorenzo_3d1l<T, Eq, T>
        <<<x_lorenzo<3>::thread_grid(len3), x_lorenzo<3>::thread_block, 0,
           (cudaStream_t)stream>>>(
            eq, outlier, len3, leap3, radius, ebx2, xdata);
  }

  STOP_GPUEVENT_RECORDING((cudaStream_t)stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}

#define CPP_INS(T, Eq, TIMING)                                             \
  template pszerror pszcxx_reverse_predict_lorenzo<T, Eq, TIMING>(         \
      Eq * eq, dim3 const len3, T* outlier, f8 const eb, int const radius, \
      T* xdata, f4* time_elapsed, void* stream);

// CPP_INS(f4, u1);
// CPP_INS(f8, u1);

// CPP_INS(f4, u2);
// CPP_INS(f8, u2);

CPP_INS(f4, u4, true);
CPP_INS(f8, u4, true);
CPP_INS(f4, u4, false);
CPP_INS(f8, u4, false);

#undef CPP_INS
