/**
 * @file lproto.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-09-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "cusz/type.h"
#include "detail/lproto.inl"
#include "kernel/lproto.hh"
#include "mem/compact.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

template <typename T, typename Eq>
pszerror psz_comp_lproto(
    T* const data, dim3 const len3, double const eb, int const radius,
    Eq* const eq, void* _outlier, float* time_elapsed, cudaStream_t stream)
{
  auto divide3 = [](dim3 len, dim3 sublen) {
    return dim3(
        (len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1,
        (len.z - 1) / sublen.z + 1);
  };

  auto ndim = [&]() {
    if (len3.z == 1 and len3.y == 1)
      return 1;
    else if (len3.z == 1 and len3.y != 1)
      return 2;
    else
      return 3;
  };

  auto outlier = (CompactGpuDram<T>*)_outlier;

  constexpr auto Tile1D = 256;
  constexpr auto Block1D = dim3(256, 1, 1);
  auto Grid1D = divide3(len3, Tile1D);

  constexpr auto Tile2D = dim3(16, 16, 1);
  constexpr auto Block2D = dim3(16, 16, 1);
  auto Grid2D = divide3(len3, Tile2D);

  constexpr auto Tile3D = dim3(8, 8, 8);
  constexpr auto Block3D = dim3(8, 8, 8);
  auto Grid3D = divide3(len3, Tile3D);

  // error bound
  auto ebx2 = eb * 2;
  auto ebx2_r = 1 / ebx2;
  auto leap3 = dim3(1, len3.x, len3.x * len3.y);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  using namespace psz::cuda_hip::__kernel::proto;

  if (ndim() == 1) {
    c_lorenzo_1d1l<T, Eq><<<Grid1D, Block1D, 0, stream>>>(
        data, len3, leap3, radius, ebx2_r, eq, *outlier);
  }
  else if (ndim() == 2) {
    c_lorenzo_2d1l<T, Eq><<<Grid2D, Block2D, 0, stream>>>(
        data, len3, leap3, radius, ebx2_r, eq, *outlier);
  }
  else if (ndim() == 3) {
    c_lorenzo_3d1l<T, Eq><<<Grid3D, Block3D, 0, stream>>>(
        data, len3, leap3, radius, ebx2_r, eq, *outlier);
  }
  else {
    throw std::runtime_error("Lorenzo only works for 123-D.");
  }

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize(stream));

  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}

template <typename T, typename Eq>
pszerror psz_decomp_lproto(
    Eq* eq, dim3 const len3, T* scattered_outlier, double const eb,
    int const radius, T* xdata, float* time_elapsed, cudaStream_t stream)
{
  auto divide3 = [](dim3 len, dim3 sublen) {
    return dim3(
        (len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1,
        (len.z - 1) / sublen.z + 1);
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
  constexpr auto Block1D = dim3(256, 1, 1);
  auto Grid1D = divide3(len3, Tile1D);

  constexpr auto Tile2D = dim3(16, 16, 1);
  constexpr auto Block2D = dim3(16, 16, 1);
  auto Grid2D = divide3(len3, Tile2D);

  constexpr auto Tile3D = dim3(8, 8, 8);
  constexpr auto Block3D = dim3(8, 8, 8);
  auto Grid3D = divide3(len3, Tile3D);

  // error bound
  auto ebx2 = eb * 2;
  auto ebx2_r = 1 / ebx2;
  auto leap3 = dim3(1, len3.x, len3.x * len3.y);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  using namespace psz::cuda_hip::__kernel::proto;

  if (ndim() == 1) {
    x_lorenzo_1d1l<T, Eq><<<Grid1D, Block1D, 0, stream>>>(
        eq, scattered_outlier, len3, leap3, radius, ebx2, xdata);
  }
  else if (ndim() == 2) {
    x_lorenzo_2d1l<T, Eq><<<Grid2D, Block2D, 0, stream>>>(
        eq, scattered_outlier, len3, leap3, radius, ebx2, xdata);
  }
  else if (ndim() == 3) {
    x_lorenzo_3d1l<T, Eq><<<Grid3D, Block3D, 0, stream>>>(
        eq, scattered_outlier, len3, leap3, radius, ebx2, xdata);
  }

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize(stream));

  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}

#define CPP_INS(T, Eq)                                                 \
  template pszerror psz_comp_lproto<T, Eq>(                   \
      T* const, dim3 const, double const, int const, Eq* const, void*, \
      float*, cudaStream_t);                                           \
                                                                       \
  template pszerror psz_decomp_lproto<T, Eq>(                 \
      Eq*, dim3 const, T*, double const, int const, T*, float*,        \
      cudaStream_t);

// TODO decrease the number of instantiated types
CPP_INS(float, uint8_t);
CPP_INS(float, uint16_t);
CPP_INS(float, uint32_t);
// CPP_INS(float, float);
// CPP_INS(float, int32_t);

CPP_INS(double, uint8_t);
CPP_INS(double, uint16_t);
CPP_INS(double, uint32_t);
// CPP_INS(double, float);
// CPP_INS(double, int32_t);

#undef CPP_INS
