// revived; updated with compact_outlier v2

#include <cuda_runtime.h>

#include "kernel.hh"
#include "mem/cxx_backends.h"
#include "spl_y24.cuh"

template <typename T, typename E, typename FP>
int psz::module::GPU_x_spline_y24<T, E, FP>::kernel_v1(
    host::view<T> anchor, host::view<E> eq, host::view<T> xdata, T*, double eb, uint32_t radius,
    INTERP_PARAMS, void* stream)
{
  auto extent = LEN_TO_DIM3(xdata.extent);
  if (extent.z == 1) return PSZ_ABORT_UNSUPPORTED_DIMENSION;  // 3D-only

  constexpr int BLK8 = 8;
  auto div = [](auto a, auto b) { return (a - 1) / b + 1; };
  auto data_leap = LEN_TO_DIM3(xdata.leap);
  auto anchor_leap = LEN_TO_DIM3(anchor.leap);
  auto grid = dim3(div(extent.x, BLK8 * 4), div(extent.y, BLK8), div(extent.z, BLK8));
  auto ebx2 = (FP)(eb * 2), eb_r = (FP)(1 / eb);

  psz::KCU_x_spline3d_infprecis_32x8x8data<E, T, FP, DEFAULT_LINEAR_BLOCK_SIZE>  //
      <<<grid, dim3(DEFAULT_LINEAR_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
          eq.ptr, extent, data_leap, anchor.ptr, LEN_TO_DIM3(anchor.extent), anchor_leap,
          xdata.ptr, extent, data_leap, eb_r, ebx2, radius);

  return CUSZ_SUCCESS;
}
