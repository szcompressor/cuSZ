// Authors: Jinyang Liu, Shixun Wu, Jiannan Tian

#include <cuda_runtime.h>

#include "kernel.hh"
#include "mem/cxx_backends.h"
#include "spl_y25.cuh"

constexpr int DEFAULT_BLOCK_SIZE = BLOCK_DIM_SIZE;
constexpr int LEVEL = 6;
constexpr int SPLINE_DIM_2 = 2;
constexpr int SPLINE_DIM_3 = 3;
constexpr int AnchorBlockSizeX = 64;
constexpr int AnchorBlockSizeY = 64;
constexpr int AnchorBlockSizeZ = 1;
constexpr int NumAnchorBlockX = 1;
constexpr int NumAnchorBlockY = 1;
constexpr int NumAnchorBlockZ = 1;
constexpr int PROFILE_BLOCK_SIZE_X = 4;
constexpr int PROFILE_BLOCK_SIZE_Y = 4;
constexpr int PROFILE_BLOCK_SIZE_Z = 4;
constexpr int PROFILE_NUM_BLOCK_X = 4;
constexpr int PROFILE_NUM_BLOCK_Y = 4;
constexpr int PROFILE_NUM_BLOCK_Z = 4;

template <typename T, typename E, typename FP>
int psz::module::GPU_x_spline_y25<T, E, FP>::kernel_v1(
    host::view<T> anchor, host::view<E> ectrl, host::view<T> xdata, T* outlier_tmp, double eb,
    uint32_t radius, INTERP_PARAMS intp_param, void* stream)
{
  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  auto ebx2 = eb * 2;
  auto eb_r = 1 / eb;

  auto l3 = LEN_TO_DIM3(xdata.extent);
  auto data_stride3 = LEN_TO_DIM3(xdata.leap);
  auto anchor_l3 = LEN_TO_DIM3(anchor.extent);
  auto anchor_stride3 = LEN_TO_DIM3(anchor.leap);
  auto extent = l3;

  if (l3.z == 1) {
    auto grid_dim = dim3(
        div(extent.x, AnchorBlockSizeX * NumAnchorBlockX),
        div(extent.y, AnchorBlockSizeY * NumAnchorBlockY),
        div(extent.z, AnchorBlockSizeZ * NumAnchorBlockZ));
    psz::KCU_x_spl_infprecis_data<
        E, T, FP, LEVEL, SPLINE_DIM_2, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        NumAnchorBlockX, NumAnchorBlockY, NumAnchorBlockZ, DEFAULT_BLOCK_SIZE>  //
        <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
            ectrl.ptr, extent, data_stride3, anchor.ptr, anchor_l3, anchor_stride3,
            xdata.ptr, extent, data_stride3, xdata.ptr, eb_r, ebx2, radius,
            intp_param);
  }
  else {
    auto grid_dim =
        dim3(div(extent.x, BLOCK16), div(extent.y, BLOCK16), div(extent.z, BLOCK16));
    psz::KCU_x_spl_infprecis_data<
        E, T, FP, 4, SPLINE_DIM_3, BLOCK16, BLOCK16, BLOCK16, 1, 1, 1, DEFAULT_BLOCK_SIZE>  //
        <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
            ectrl.ptr, extent, data_stride3, anchor.ptr, anchor_l3, anchor_stride3,
            xdata.ptr, extent, data_stride3, xdata.ptr, eb_r, ebx2, radius,
            intp_param);
  }

  cudaStreamSynchronize((cudaStream_t)stream);
  // TIME_ELAPSED_GPUEVENT(time);

  return 0;
}
