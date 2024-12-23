/**
 * @file lorenzo_proto.inl
 * @author Jiannan Tian
 * @brief (prototype) Dual-Eq Lorenzo method.
 * @version 0.2
 * @date 2019-09-23
 * (create) 2019-09-23 (rev) 2023-04-03
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory See LICENSE in top-level directory
 *
 */

#include <cstddef>
#include <stdexcept>

#include "mem/cxx_sp_gpu.h"
#include "utils/err.hh"
#include "utils/it_cuda.hh"
#include "utils/timer.hh"

// easy algorithmic description
namespace psz {

template <
    typename T, int TileDim = 256, typename Eq = uint16_t, typename Fp = T>
__global__ void KERNEL_CUHIP_prototype_x_lorenzo_1d1l(
    Eq* const in_eq, T* const in_outlier, T* const out_data,
    dim3 const data_len3, dim3 const data_leap3, uint16_t const radius,
    Fp const ebx2)
{
  SETUP_ND_GPU_CUDA;
  __shared__ T buf[TileDim];

  auto id = gid1();
  auto data = [&](auto dx) -> T& { return buf[t().x + dx]; };

  if (id < data_len3.x)
    data(0) = in_outlier[id] + static_cast<T>(in_eq[id]) - radius;  // fuse
  else
    data(0) = 0;
  __syncthreads();

  for (auto d = 1; d < TileDim; d *= 2) {
    T n = 0;
    if (t().x >= d)
      n = data(-d);  // like __shfl_up_sync(0x1f, var, d); warp_sync
    __syncthreads();
    if (t().x >= d) data(0) += n;
    __syncthreads();
  }

  if (id < data_len3.x) { out_data[id] = data(0) * ebx2; }
}

template <
    typename T, int TileDim = 16, typename Eq = uint16_t, typename Fp = T>
__global__ void KERNEL_CUHIP_prototype_x_lorenzo_2d1l(
    Eq* const in_eq, T* const in_outlier, T* const out_data,
    dim3 const data_len3, dim3 const data_leap3, uint16_t const radius,
    Fp const ebx2)
{
  SETUP_ND_GPU_CUDA;
  __shared__ T buf[TileDim][TileDim + 1];

  auto id = gid2();
  auto data = [&](auto dx, auto dy) -> T& {
    return buf[t().y + dy][t().x + dx];
  };

  if (check_boundary2())
    data(0, 0) = in_outlier[id] + static_cast<T>(in_eq[id]) - radius;  // fuse
  else
    data(0, 0) = 0;
  __syncthreads();

  for (auto d = 1; d < TileDim; d *= 2) {
    T n = 0;
    if (t().x >= d) n = data(-d, 0);
    __syncthreads();
    if (t().x >= d) data(0, 0) += n;
    __syncthreads();
  }

  for (auto d = 1; d < TileDim; d *= 2) {
    T n = 0;
    if (t().y >= d) n = data(0, -d);
    __syncthreads();
    if (t().y >= d) data(0, 0) += n;
    __syncthreads();
  }

  if (check_boundary2()) { out_data[id] = data(0, 0) * ebx2; }
}

template <typename T, int TileDim = 8, typename Eq = uint16_t, typename Fp = T>
__global__ void KERNEL_CUHIP_prototype_x_lorenzo_3d1l(
    Eq* const in_eq, T* const in_outlier, T* const out_data,
    dim3 const data_len3, dim3 const data_leap3, uint16_t const radius,
    Fp const ebx2)
{
  SETUP_ND_GPU_CUDA;
  __shared__ T buf[TileDim][TileDim][TileDim + 1];

  auto id = gid3();
  auto data = [&](auto dx, auto dy, auto dz) -> T& {
    return buf[t().z + dz][t().y + dy][t().x + dx];
  };

  if (check_boundary3())
    data(0, 0, 0) = in_outlier[id] + static_cast<T>(in_eq[id]) - radius;
  else
    data(0, 0, 0) = 0;
  __syncthreads();

  for (auto dist = 1; dist < TileDim; dist *= 2) {
    T addend = 0;
    if (t().x >= dist) addend = data(-dist, 0, 0);
    __syncthreads();
    if (t().x >= dist) data(0, 0, 0) += addend;
    __syncthreads();
  }

  for (auto dist = 1; dist < TileDim; dist *= 2) {
    T addend = 0;
    if (t().y >= dist) addend = data(0, -dist, 0);
    __syncthreads();
    if (t().y >= dist) data(0, 0, 0) += addend;
    __syncthreads();
  }

  for (auto dist = 1; dist < TileDim; dist *= 2) {
    T addend = 0;
    if (t().z >= dist) addend = data(0, 0, -dist);
    __syncthreads();
    if (t().z >= dist) data(0, 0, 0) += addend;
    __syncthreads();
  }

  if (check_boundary3()) { out_data[id] = data(0, 0, 0) * ebx2; }
}

}  // namespace psz

namespace psz::cuhip {

template <typename T, typename Eq = uint16_t>
pszerror GPU_PROTO_x_lorenzo_nd(
    Eq* in_eq, T* in_outlier, T* out_data, dim3 const data_len3,
    double const eb, int const radius, float* time_elapsed, void* stream)
{
  auto divide3 = [](dim3 len, dim3 sublen) {
    return dim3(
        (len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1,
        (len.z - 1) / sublen.z + 1);
  };

  auto ndim = [&]() {
    if (data_len3.z == 1 and data_len3.y == 1)
      return 1;
    else if (data_len3.z == 1 and data_len3.y != 1)
      return 2;
    else
      return 3;
  };

  constexpr auto Tile1D = dim3(256, 1, 1), Tile2D = dim3(16, 16, 1),
                 Tile3D = dim3(8, 8, 8);
  constexpr auto Block1D = dim3(256, 1, 1), Block2D = dim3(16, 16, 1),
                 Block3D = dim3(8, 8, 8);

  auto Grid1D = divide3(data_len3, Tile1D),
       Grid2D = divide3(data_len3, Tile2D),
       Grid3D = divide3(data_len3, Tile3D);

  // error bound
  auto ebx2 = eb * 2, ebx2_r = 1 / ebx2;
  auto data_leap3 = dim3(1, data_len3.x, data_len3.x * data_len3.y);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  if (ndim() == 1) {
    psz::KERNEL_CUHIP_prototype_x_lorenzo_1d1l<T>
        <<<Grid1D, Block1D, 0, (cudaStream_t)stream>>>(
            in_eq, in_outlier, out_data, data_len3, data_leap3, radius, ebx2);
  }
  else if (ndim() == 2) {
    psz::KERNEL_CUHIP_prototype_x_lorenzo_2d1l<T>
        <<<Grid2D, Block2D, 0, (cudaStream_t)stream>>>(
            in_eq, in_outlier, out_data, data_len3, data_leap3, radius, ebx2);
  }
  else if (ndim() == 3) {
    psz::KERNEL_CUHIP_prototype_x_lorenzo_3d1l<T>
        <<<Grid3D, Block3D, 0, (cudaStream_t)stream>>>(
            in_eq, in_outlier, out_data, data_len3, data_leap3, radius, ebx2);
  }

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}

}  // namespace psz::cuhip

////////////////////////////////////////////////////////////////////////////////
#define INSTANTIATIE_GPU_LORENZO_PROTO_X_2params(T)                      \
  template pszerror psz::cuhip::GPU_PROTO_x_lorenzo_nd<T>(               \
      uint16_t* in_eq, T* in_outlier, T* out_data, dim3 const data_len3, \
      double const eb, int const radius, float* time_elapsed, void* stream);

#define INSTANTIATIE_LORENZO_PROTO_X_1param(T) \
  INSTANTIATIE_GPU_LORENZO_PROTO_X_2params(T);
