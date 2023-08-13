/**
 * @file l32r.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-04-04
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <type_traits>

#include "cusz/type.h"
#include "detail/l23r.inl"
#include "kernel/l23r.hh"
#include "pipeline/compact_cuda.inl"
#include "utils/cuda_err.cuh"
#include "utils/timer.h"

template <typename T, bool UsePnEnc, typename Eq>
cusz_error_status psz_comp_l23r(
    T* const data, dim3 const len3, double const eb, int const radius,
    Eq* const eq, void* _outlier, float* time_elapsed, cudaStream_t stream)
{
  static_assert(
      std::is_same<Eq, uint32_t>::value or std::is_same<Eq, uint16_t>::value or
          std::is_same<Eq, uint8_t>::value,
      "Eq must be unsigned integer that is less than or equal to 4 bytes.");

  auto divide3 = [](dim3 len, dim3 tile) {
    return dim3(
        (len.x - 1) / tile.x + 1, (len.y - 1) / tile.y + 1,
        (len.z - 1) / tile.z + 1);
  };

  auto ndim = [&]() {
    if (len3.z == 1 and len3.y == 1)
      return 1;
    else if (len3.z == 1 and len3.y != 1)
      return 2;
    else
      return 3;
  };

  using Compact = CompactCudaDram<T>;
  auto ot = (Compact*)_outlier;

  constexpr auto Tile1D = 256;
  constexpr auto Seq1D = 4;
  constexpr auto Block1D = 64;
  auto Grid1D = divide3(len3, Tile1D);

  constexpr auto Tile2D = dim3(16, 16, 1);
  constexpr auto Block2D = dim3(16, 2, 1);
  auto Grid2D = divide3(len3, Tile2D);

  constexpr auto Tile3D = dim3(32, 8, 8);
  constexpr auto Block3D = dim3(32, 8, 1);
  auto Grid3D = divide3(len3, Tile3D);

  auto d = ndim();

  // error bound
  auto ebx2 = eb * 2;
  auto ebx2_r = 1 / ebx2;
  auto leap3 = dim3(1, len3.x, len3.x * len3.y);

  CREATE_CUDAEVENT_PAIR;
  START_CUDAEVENT_RECORDING(stream);

  if (d == 1) {
    psz::rolling::c_lorenzo_1d1l<T, false, Eq, T, Tile1D, Seq1D>
        <<<Grid1D, Block1D, 0, stream>>>(
            data, len3, leap3, radius, ebx2_r, eq, ot->val, ot->idx, ot->num);
  }
  else if (d == 2) {
    psz::rolling::c_lorenzo_2d1l<T, false, Eq, T>
        <<<Grid2D, Block2D, 0, stream>>>(
            data, len3, leap3, radius, ebx2_r, eq, ot->val, ot->idx, ot->num);
  }
  else if (d == 3) {
    psz::rolling::c_lorenzo_3d1l<T, false, Eq, T>
        <<<Grid3D, Block3D, 0, stream>>>(
            data, len3, leap3, radius, ebx2_r, eq, ot->val, ot->idx, ot->num);
  }

  STOP_CUDAEVENT_RECORDING(stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  TIME_ELAPSED_CUDAEVENT(time_elapsed);
  DESTROY_CUDAEVENT_PAIR;

  return CUSZ_SUCCESS;
}

template cusz_error_status psz_comp_l23r<float, false>(
    float* const data, dim3 const len3, double const eb, int const radius,
    uint32_t* const eq, void* _outlier, float* time_elapsed,
    cudaStream_t stream);

template cusz_error_status psz_comp_l23r<float, true>(
    float* const data, dim3 const len3, double const eb, int const radius,
    uint32_t* const eq, void* _outlier, float* time_elapsed,
    cudaStream_t stream);

template cusz_error_status psz_comp_l23r<double, false>(
    double* const data, dim3 const len3, double const eb, int const radius,
    uint32_t* const eq, void* _outlier, float* time_elapsed,
    cudaStream_t stream);

template cusz_error_status psz_comp_l23r<double, true>(
    double* const data, dim3 const len3, double const eb, int const radius,
    uint32_t* const eq, void* _outlier, float* time_elapsed,
    cudaStream_t stream);