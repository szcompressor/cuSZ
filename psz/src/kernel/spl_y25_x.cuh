// Authors: Jinyang Liu, Shixun Wu, Jiannan Tian

#include <cuda_runtime.h>

#include "kernel.hh"
#include "mem/buf_comp.hh"
#include "mem/cxx_backends.h"
#include "spl_y25.cuh"

constexpr int LEVEL = 6;
constexpr int SplDim2 = 2;
constexpr int SplDim3 = 3;
constexpr int AncBlkSzX = 64;
constexpr int AncBlkSzY = 64;
constexpr int AncBlkSzZ = 1;
constexpr int NAncBlkX = 1;
constexpr int NAncBlkY = 1;
constexpr int NAncBlkZ = 1;
constexpr int ProfBlkSzX = 4;
constexpr int ProfBlkSzY = 4;
constexpr int ProfBlkSzZ = 4;
constexpr int ProfNBlkX = 4;
constexpr int ProfNBlkY = 4;
constexpr int ProfNBlkZ = 4;

template <class Types>
int psz::module::GPU_x_spline_y25<Types>::kernel(
    Buf* buf, T* anchor_p, host::view<T> xdata, double eb, uint32_t radius,
    INTERP_PARAMS intp_param, void* stream)
{
  using FP = typename Types::Fp;
  auto anchor = _ptb::make_view(anchor_p, buf->anchor_len3());
  auto eq = _ptb::make_view(buf->eq_d(), xdata.extent);
  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  auto ebx2 = eb * 2;
  auto eb_r = 1 / eb;

  auto l3 = LEN_TO_DIM3(xdata.extent);
  auto data_leap = LEN_TO_DIM3(xdata.leap);
  auto anchor_l3 = LEN_TO_DIM3(anchor.extent);
  auto anchor_leap = LEN_TO_DIM3(anchor.leap);
  auto extent = l3;

  if (l3.z == 1) {
    auto grid_dim = dim3(
        div(extent.x, AncBlkSzX * NAncBlkX), div(extent.y, AncBlkSzY * NAncBlkY),
        div(extent.z, AncBlkSzZ * NAncBlkZ));
    psz::KCU_x_spl_infprecis_data<
        E, T, FP, LEVEL, SplDim2, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
        DefaultLinBlkSz>  //
        <<<grid_dim, dim3(DefaultLinBlkSz, 1, 1), 0, (cudaStream_t)stream>>>(
            eq.ptr, extent, data_leap, anchor.ptr, anchor_l3, anchor_leap, xdata.ptr, extent,
            data_leap, xdata.ptr, eb_r, ebx2, radius, intp_param);
  }
  else {
    auto grid_dim = dim3(div(extent.x, Blk16), div(extent.y, Blk16), div(extent.z, Blk16));
    psz::KCU_x_spl_infprecis_data<
        E, T, FP, 4, SplDim3, Blk16, Blk16, Blk16, 1, 1, 1, DefaultLinBlkSz>  //
        <<<grid_dim, dim3(DefaultLinBlkSz, 1, 1), 0, (cudaStream_t)stream>>>(
            eq.ptr, extent, data_leap, anchor.ptr, anchor_l3, anchor_leap, xdata.ptr, extent,
            data_leap, xdata.ptr, eb_r, ebx2, radius, intp_param);
  }

  cudaStreamSynchronize((cudaStream_t)stream);
  // TIME_ELAPSED_GPUEVENT(time);

  return 0;
}
