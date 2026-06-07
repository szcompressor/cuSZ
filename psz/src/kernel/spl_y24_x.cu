// revived; updated with compact_outlier v2

#include <cuda_runtime.h>

#include "kernel.hh"
#include "mem/cxx_backends.h"
#include "spl_y24.cuh"

template <typename T, typename E, typename FP>
int psz::module::GPU_x_spline_y24<T, E, FP>::kernel_v1(
    T* anchor, psz_len const anchor_len3, E* ectrl, T* xdata, psz_len const xdata_len3, T*,
    double eb, uint32_t radius, INTERP_PARAMS, void* stream)
{
  auto l3 = LEN_TO_DIM3(xdata_len3);
  if (l3.z == 1) return PSZ_ABORT_UNSUPPORTED_DIMENSION;  // 3D-only

  constexpr int BLK8 = 8;
  auto div = [](auto a, auto b) { return (a - 1) / b + 1; };
  auto grid = dim3(div(l3.x, BLK8 * 4), div(l3.y, BLK8), div(l3.z, BLK8));
  auto ebx2 = (FP)(eb * 2), eb_r = (FP)(1 / eb);

  psz::KCU_x_spline3d_infprecis_32x8x8data<E, T, FP, DEFAULT_LINEAR_BLOCK_SIZE>  //
      <<<grid, dim3(DEFAULT_LINEAR_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
          ectrl, LEN_TO_DIM3(xdata_len3), LEN_TO_STRIDE3(xdata_len3),     //
          anchor, LEN_TO_DIM3(anchor_len3), LEN_TO_STRIDE3(anchor_len3),  //
          xdata, LEN_TO_DIM3(xdata_len3), LEN_TO_STRIDE3(xdata_len3),     //
          eb_r, ebx2, radius);

  return CUSZ_SUCCESS;
}
