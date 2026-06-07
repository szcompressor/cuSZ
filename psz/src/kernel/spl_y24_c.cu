// revived; updated with compact_outlier v2

#include <cuda_runtime.h>

#include "kernel.hh"
#include "mem/cxx_backends.h"
#include "mem/cxx_sp_gpu.h"
#include "spl_y24.cuh"

template <typename T, typename E, typename FP>
int psz::module::GPU_c_spline_y24<T, E, FP>::kernel_v1(
    host::view<T> data, host::view<E> eq, host::view<T> anchor, void* _outlier, double eb, double,
    uint32_t radius, INTERP_PARAMS&, T*, T*, u4 const, void* stream)
{
  auto extent = LEN_TO_DIM3(data.extent);
  if (extent.z == 1) return PSZ_ABORT_UNSUPPORTED_DIMENSION;  // 3D-only

  constexpr int BLK8 = 8;
  auto div = [](auto a, auto b) { return (a - 1) / b + 1; };
  auto data_leap = LEN_TO_DIM3(data.leap);
  auto anchor_leap = LEN_TO_DIM3(anchor.leap);
  auto grid = dim3(div(extent.x, BLK8 * 4), div(extent.y, BLK8), div(extent.z, BLK8));
  auto ebx2 = (FP)(eb * 2), eb_r = (FP)(1 / eb);

  using Compact2 = _ptb::compact_GPU_DRAM2<T, u4>;
  using Cell = _ptb::compact_cell<T, u4>;
  auto ot = (Compact2*)_outlier;

  psz::KCU_c_spline3d_infprecis_32x8x8data<T, E, FP, DEFAULT_LINEAR_BLOCK_SIZE, Cell*>  //
      <<<grid, dim3(DEFAULT_LINEAR_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
          data.ptr, extent, data_leap, eq.ptr, extent, data_leap, anchor.ptr, anchor_leap,
          ot->val_idx_d(), ot->num_d(), eb_r, ebx2, radius);

  return CUSZ_SUCCESS;
}
