// revived; updated with compact_outlier v2

#include <cuda_runtime.h>

#include "kernel.hh"
#include "mem/cxx_backends.h"
#include "mem/cxx_sp_gpu.h"
#include "spl_y24.cuh"

template <typename T, typename E, typename FP>
int psz::module::GPU_c_spline_y24<T, E, FP>::kernel_v1(
    T* data, psz_len const data_len3, T* anchor, psz_len const anchor_len3, E* ectrl,
    void* _outlier, double eb, double, uint32_t radius, INTERP_PARAMS&, T*, T*, u4 const,
    void* stream)
{
  auto l3 = LEN_TO_DIM3(data_len3);
  if (l3.z == 1) return PSZ_ABORT_UNSUPPORTED_DIMENSION;  // 3D-only

  constexpr int BLK8 = 8;
  auto div = [](auto a, auto b) { return (a - 1) / b + 1; };
  auto grid = dim3(div(l3.x, BLK8 * 4), div(l3.y, BLK8), div(l3.z, BLK8));
  auto ebx2 = (FP)(eb * 2), eb_r = (FP)(1 / eb);

  using Compact2 = _portable::compact_GPU_DRAM2<T, u4>;
  using Cell = _portable::compact_cell<T, u4>;
  auto ot = (Compact2*)_outlier;

  psz::KCU_c_spline3d_infprecis_32x8x8data<T, E, FP, DEFAULT_LINEAR_BLOCK_SIZE, Cell*>  //
      <<<grid, dim3(DEFAULT_LINEAR_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
          data, LEN_TO_DIM3(data_len3), LEN_TO_STRIDE3(data_len3),   //
          ectrl, LEN_TO_DIM3(data_len3), LEN_TO_STRIDE3(data_len3),  //
          anchor, LEN_TO_STRIDE3(anchor_len3),                       //
          ot->val_idx_d(), ot->num_d(), eb_r, ebx2, radius);

  return CUSZ_SUCCESS;
}
