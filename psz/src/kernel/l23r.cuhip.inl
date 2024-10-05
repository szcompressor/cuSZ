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
#include "detail/l23r.cuhip.inl"

#define LAUNCH_KERNEL_L23R                                                 \
  if (d == 1) {                                                            \
    psz::rolling::KERNEL_CUHIP_c_lorenzo_1d1l<                             \
        T, E, T, c_lorenzo<1>::tile.x, c_lorenzo<1>::sequentiality.x>      \
        <<<c_lorenzo<1>::thread_grid(len3), c_lorenzo<1>::thread_block, 0, \
           (cudaStream_t)stream>>>(                                        \
            data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),   \
            ot->num());                                                    \
  }                                                                        \
  else if (d == 2) {                                                       \
    psz::rolling::KERNEL_CUHIP_c_lorenzo_2d1l<T, E, T>                     \
        <<<c_lorenzo<2>::thread_grid(len3), c_lorenzo<2>::thread_block, 0, \
           (cudaStream_t)stream>>>(                                        \
            data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),   \
            ot->num());                                                    \
  }                                                                        \
  else if (d == 3) {                                                       \
    psz::rolling::KERNEL_CUHIP_c_lorenzo_3d1l<T, E, T>                     \
        <<<c_lorenzo<3>::thread_grid(len3), c_lorenzo<3>::thread_block, 0, \
           (cudaStream_t)stream>>>(                                        \
            data, len3, leap3, radius, ebx2_r, eq, ot->val(), ot->idx(),   \
            ot->num());                                                    \
  }

#define LAUNCH_KERNEL_L23_PREQUANT                                            \
  psz::rolling::KERNEL_CUHIP_lorenzo_prequant<                                \
      TIN, TOUT, ReverseProcess, TIN, c_lorenzo<1>::tile.x,                   \
      c_lorenzo<1>::sequentiality.x>                                          \
      <<<c_lorenzo<1>::thread_grid(dim3(len)), c_lorenzo<1>::thread_block, 0, \
         (cudaStream_t)stream>>>(in, len, ebx2_r, ebx2, out);

namespace psz::cuhip {

template <typename T, typename E, psz_timing_mode TIMING, bool ZigZag>
pszerror GPU_c_lorenzo_nd_with_outlier(
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

  if constexpr (TIMING == SYNC_BY_STREAM) {
    CREATE_GPUEVENT_PAIR;
    START_GPUEVENT_RECORDING((cudaStream_t)stream);

    LAUNCH_KERNEL_L23R;

    STOP_GPUEVENT_RECORDING((cudaStream_t)stream);
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
    TIME_ELAPSED_GPUEVENT(time_elapsed);
    DESTROY_GPUEVENT_PAIR;
  }
  else if constexpr (TIMING == CPU_BARRIER) {
    LAUNCH_KERNEL_L23R;
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  }
  else if constexpr (TIMING == GPU_AUTOMONY) {
    LAUNCH_KERNEL_L23R;
  }
  else {
    return PSZ_WRONG_TIMER_SPECIFIED;
  }

  return CUSZ_SUCCESS;
}

template <
    typename TIN, typename TOUT, bool ReverseProcess, psz_timing_mode TIMING>
pszerror GPU_lorenzo_prequant(
    TIN* const in, size_t const len, PROPER_EB const eb, TOUT* const out,
    float* time_elapsed, void* stream)
{
  using namespace psz::kernelconfig;
  // error bound
  auto ebx2 = eb * 2, ebx2_r = 1 / ebx2;

  if constexpr (TIMING == SYNC_BY_STREAM) {
    CREATE_GPUEVENT_PAIR;
    START_GPUEVENT_RECORDING((cudaStream_t)stream);

    LAUNCH_KERNEL_L23_PREQUANT;

    STOP_GPUEVENT_RECORDING((cudaStream_t)stream);
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
    TIME_ELAPSED_GPUEVENT(time_elapsed);
    DESTROY_GPUEVENT_PAIR;
  }
  else if constexpr (TIMING == CPU_BARRIER) {
    LAUNCH_KERNEL_L23_PREQUANT;
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  }
  else if constexpr (TIMING == GPU_AUTOMONY) {
    LAUNCH_KERNEL_L23_PREQUANT;
  }
  else {
    return PSZ_WRONG_TIMER_SPECIFIED;
  }

  return CUSZ_SUCCESS;
}

}  // namespace psz::cuhip

// -----------------------------------------------------------------------------
#define INSTANCIATE_GPU_L23R_4params(T, E, TIMING, ZIGZAG)           \
  template pszerror                                                  \
  psz::cuhip::GPU_c_lorenzo_nd_with_outlier<T, E, TIMING, ZIGZAG>(   \
      T* const data, dim3 const len3, f8 const eb, int const radius, \
      E* const eq, void* _outlier, f4* time_elapsed, void* stream);

#define INSTANCIATE_GPU_L23R_3params(T, E, TIMING)   \
  INSTANCIATE_GPU_L23R_4params(T, E, TIMING, false); \
  INSTANCIATE_GPU_L23R_4params(T, E, TIMING, true);

#define INSTANCIATE_GPU_L23R_2params(T, E)            \
  INSTANCIATE_GPU_L23R_3params(T, E, SYNC_BY_STREAM); \
  INSTANCIATE_GPU_L23R_3params(T, E, CPU_BARRIER);    \
  INSTANCIATE_GPU_L23R_3params(T, E, GPU_AUTOMONY);

#define INSTANCIATE_GPU_L23R_1param(T) \
  INSTANCIATE_GPU_L23R_2params(T, u2); \
  INSTANCIATE_GPU_L23R_2params(T, u4);
// -----------------------------------------------------------------------------
#define INSTANCIATE_GPU_L23_PREQ_4params(TIN, TOUT, REV, TIMING)              \
  template pszerror psz::cuhip::GPU_lorenzo_prequant<TIN, TOUT, REV, TIMING>( \
      TIN* const in, size_t const len, PROPER_EB const eb, TOUT* const out,   \
      float* time_elapsed, void* stream);

#define INSTANCIATE_GPU_L23_PREQ_3params(TIN, REV, TIMING)     \
  INSTANCIATE_GPU_L23_PREQ_4params(TIN, int32_t, REV, TIMING); \
  INSTANCIATE_GPU_L23_PREQ_4params(TIN, int16_t, REV, TIMING); \
  INSTANCIATE_GPU_L23_PREQ_4params(TIN, int8_t, REV, TIMING);

#define INSTANCIATE_GPU_L23_PREQ_2params(T, REV)            \
  INSTANCIATE_GPU_L23_PREQ_3params(T, REV, SYNC_BY_STREAM); \
  INSTANCIATE_GPU_L23_PREQ_3params(T, REV, CPU_BARRIER);    \
  INSTANCIATE_GPU_L23_PREQ_3params(T, REV, GPU_AUTOMONY);

#define INSTANCIATE_GPU_L23_PREQ_1param(T)    \
  INSTANCIATE_GPU_L23_PREQ_2params(T, false); \
  INSTANCIATE_GPU_L23_PREQ_2params(T, true);

//  -----------------------------------------------------------------------------

#undef LAUNCH_KERNEL_L23R
#undef LAUNCH_KERNEL_L23_PREQUANT
