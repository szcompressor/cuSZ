/**
 * @file l23.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-01
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <cuda_runtime.h>

#include "cusz/type.h"
#include "detail/l23_c.cu_hip.inl"
#include "kernel/lrz.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

template <typename T, typename Eq, typename FP>
pszerror psz_comp_l23(
    T* const data, dim3 const len3, f8 const eb, int const radius,
    Eq* const eq, T* const outlier, f4* time_elapsed, void* stream)
{
  auto divide3 = [](dim3 len, dim3 subl) {
    return dim3(
        (len.x - 1) / subl.x + 1,  //
        (len.y - 1) / subl.y + 1,  //
        (len.z - 1) / subl.z + 1);
  };

  auto ndim = [&]() {
    if (len3.z == 1 and len3.y == 1)
      return 1;
    else if (len3.z == 1 and len3.y != 1)
      return 2;
    else
      return 3;
  };

  constexpr auto Tile1D = 256;
  constexpr auto Seq1D = 4;  // x-sequentiality == 4
  constexpr auto Block1D = dim3(256 / 4, 1, 1);
  auto Grid1D = divide3(len3, Tile1D);

  constexpr auto Tile2D = dim3(16, 16, 1);
  // constexpr auto Seq2D    = dim3(1, 8, 1);  // y-sequentiality == 8
  constexpr auto Block2D = dim3(16, 2, 1);
  auto Grid2D = divide3(len3, Tile2D);

  constexpr auto Tile3D = dim3(32, 8, 8);
  // constexpr auto Seq3D    = dim3(1, 8, 1);  // y-sequentiality == 8
  // constexpr auto Block3D = dim3(32, 1, 8);  // for v0
  constexpr auto Block3D = dim3(32, 8, 1);  // for v0::r1_shfl
  auto Grid3D = divide3(len3, Tile3D);

  auto d = ndim();

  // error bound
  auto ebx2 = eb * 2;
  auto ebx2_r = 1 / ebx2;
  auto leap3 = dim3(1, len3.x, len3.x * len3.y);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING((cudaStream_t)stream);

  if (d == 1) {
    psz::cuda_hip::__kernel::c_lorenzo_1d1l<T, Eq, FP, Tile1D, Seq1D>
        <<<Grid1D, Block1D, 0, (cudaStream_t)stream>>>(
            data, len3, leap3, radius, ebx2_r, eq, outlier);
  }
  else if (d == 2) {
    psz::cuda_hip::__kernel::c_lorenzo_2d1l<T, Eq, FP>
        <<<Grid2D, Block2D, 0, (cudaStream_t)stream>>>(
            data, len3, leap3, radius, ebx2_r, eq, outlier);
  }
  else if (d == 3) {
    psz::cuda_hip::__kernel::c_lorenzo_3d1l<T, Eq, FP>
        <<<Grid3D, Block3D, 0, (cudaStream_t)stream>>>(
            data, len3, leap3, radius, ebx2_r, eq, outlier);
  }

  STOP_GPUEVENT_RECORDING((cudaStream_t)stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}

#define CPP_INS(T, Eq)                                               \
  template pszerror psz_comp_l23<T, Eq>(                             \
      T* const data, dim3 const len3, f8 const eb, int const radius, \
      Eq* const eq, T* const outlier, f4* time_elapsed, void* stream);

CPP_INS(f4, u1);
CPP_INS(f4, u2);
CPP_INS(f4, u4);
CPP_INS(f4, f4);

CPP_INS(f8, u1);
CPP_INS(f8, u2);
CPP_INS(f8, u4);
CPP_INS(f8, f4);

CPP_INS(f4, i4);
CPP_INS(f8, i4);

#undef CPP_INS
