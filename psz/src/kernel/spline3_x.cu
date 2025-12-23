// Authors: Jinyang Liu, Shixun Wu, Jiannan Tian

#include <cuda_runtime.h>

#include "detail/spline3_md.inl"
#include "kernel/predictor.hh"
#include "mem/cxx_backends.h"

constexpr int DEFAULT_BLOCK_SIZE = BLOCK_DIM_SIZE;
constexpr int LEVEL = 6;
constexpr int SPLINE_DIM_2 = 2;
constexpr int SPLINE_DIM_3 = 3;
constexpr int AnchorBlockSizeX = 64;
constexpr int AnchorBlockSizeY = 64;
constexpr int AnchorBlockSizeZ = 1;
constexpr int numAnchorBlockX = 1;
constexpr int numAnchorBlockY = 1;
constexpr int numAnchorBlockZ = 1;
constexpr int PROFILE_BLOCK_SIZE_X = 4;
constexpr int PROFILE_BLOCK_SIZE_Y = 4;
constexpr int PROFILE_BLOCK_SIZE_Z = 4;
constexpr int PROFILE_NUM_BLOCK_X = 4;
constexpr int PROFILE_NUM_BLOCK_Y = 4;
constexpr int PROFILE_NUM_BLOCK_Z = 4;

template <typename T, typename E, typename FP>
int psz::module::GPU_spline_reconstruct<T, E, FP>::kernel_v1(
    T* anchor, psz_len const anchor_len3, E* ectrl, T* xdata, psz_len const xdata_len3,
    T* outlier_tmp, double eb, uint32_t radius, INTERPOLATION_PARAMS intp_param, void* stream)
{
  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  auto ebx2 = eb * 2;
  auto eb_r = 1 / eb;

  auto l3 = LEN_TO_DIM3(xdata_len3);
  auto data_stride3 = LEN_TO_STRIDE3(xdata_len3);
  auto anchor_l3 = LEN_TO_DIM3(anchor_len3);
  auto anchor_stride3 = LEN_TO_STRIDE3(anchor_len3);

  if (l3.z == 1) {
    auto grid_dim = dim3(
        div(l3.x, AnchorBlockSizeX * numAnchorBlockX),
        div(l3.y, AnchorBlockSizeY * numAnchorBlockY),
        div(l3.z, AnchorBlockSizeZ * numAnchorBlockZ));

    cusz::x_spline_infprecis_data<
        E*, T*, FP, LEVEL, SPLINE_DIM_2, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ,
        numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, DEFAULT_BLOCK_SIZE>  //
        <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
            ectrl, l3, data_stride3, anchor, anchor_l3, anchor_stride3, xdata, l3, data_stride3,
            xdata, eb_r, ebx2, radius, intp_param);
  }
  else {
    auto grid_dim = dim3(div(l3.x, BLOCK16), div(l3.y, BLOCK16), div(l3.z, BLOCK16));

    cusz::x_spline_infprecis_data<
        E*, T*, FP, 4, SPLINE_DIM_3, BLOCK16, BLOCK16, BLOCK16, 1, 1, 1, DEFAULT_BLOCK_SIZE>  //
        <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
            ectrl, l3, data_stride3, anchor, anchor_l3, anchor_stride3, xdata, l3, data_stride3,
            xdata, eb_r, ebx2, radius, intp_param);
  }

  cudaStreamSynchronize((cudaStream_t)stream);
  // TIME_ELAPSED_GPUEVENT(time);

  return 0;
}