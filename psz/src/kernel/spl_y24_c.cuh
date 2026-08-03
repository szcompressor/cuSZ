// revived; updated with compact_outlier v2

#include <cuda_runtime.h>

#include "kernel.hh"
#include "mem/buf_comp.hh"
#include "mem/cxx_backends.h"
#include "mem/cxx_sp_gpu.h"
#include "spl_y24.cuh"

template <class Types, class Features>
int psz::module::GPU_c_spline_y24<Types, Features>::kernel(
    Buf* buf, host::view<T> in, double eb, double, uint32_t radius, INTERP_PARAMS&,
    bool enable_global, void* stream)
{
  if (LEN_TO_DIM3(in.extent).z == 1) return PSZ_ABORT_UNSUPPORTED_DIMENSION;  // 3D-only

  using FP = typename Types::Fp;
  auto data_p = in.ptr;
  auto eq_p = buf->eq_d();
  auto anchor_p = buf->anchor_d();
  auto d_ext = in.extent;
  auto a_ext = buf->anchor_len3();
  auto _outlier = (void*)buf->buf_outlier2();

  auto data = _ptb::make_view(data_p, d_ext);
  auto eq = _ptb::make_view(eq_p, d_ext);
  auto anchor = _ptb::make_view(anchor_p, a_ext);
  auto extent = LEN_TO_DIM3(data.extent);

  constexpr int BLK8 = 8;
  auto div = [](auto a, auto b) { return (a - 1) / b + 1; };
  auto data_leap = LEN_TO_DIM3(data.leap);
  auto anchor_leap = LEN_TO_DIM3(anchor.leap);
  auto grid = dim3(div(extent.x, BLK8 * 4), div(extent.y, BLK8), div(extent.z, BLK8));
  auto ebx2 = (FP)(eb * 2), eb_r = (FP)(1 / eb);

  using Compact2 = _ptb::compact_GPU_DRAM2<T, u4>;
  using Cell = _ptb::compact_cell<T, u4>;
  auto ot = (Compact2*)_outlier;

  auto out_bheader = buf->buf_hf() ? (uint32_t*)buf->buf_hf()->pbk_headers_d() : nullptr;
  auto out_block_outliers = buf->block_outliers_d();
  auto go = [&](auto global_const) {
    constexpr bool Global = decltype(global_const)::value;
    using F = psz::PredictorFeature<
        Features::UseZigZag, Features::UseH1GL,
        (Global ? 0b10 : 0b00) | (Features::UnpredIncomp & 0b01)>;
    psz::KCU_c_spline3d_infprecis_32x8x8data<T, E, FP, DEFAULT_LINEAR_BLOCK_SIZE, Cell*, uint32_t*, F>
        <<<grid, dim3(DEFAULT_LINEAR_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
            data.ptr, extent, data_leap, eq.ptr, extent, data_leap, anchor.ptr, anchor_leap,
            ot->val_idx_d(), ot->num_d(), eb_r, ebx2, radius, out_bheader, out_block_outliers,
            enable_global, ot->max_allowed_num());
  };
  if (enable_global)
    go(std::integral_constant<bool, true>{});
  else
    go(std::integral_constant<bool, false>{});

  return CUSZ_SUCCESS;
}
