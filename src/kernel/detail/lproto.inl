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

#ifndef CUSZ_KERNEL_LORENZO_PROTOTYPE_CUH
#define CUSZ_KERNEL_LORENZO_PROTOTYPE_CUH

#include <cstddef>
#include <stdexcept>

#include "mem/compact.hh"
#include "utils/err.hh"
#include "utils/it_cuda.hh"
#include "utils/timer.hh"

namespace psz {

namespace cuda_hip {
namespace __kernel {

namespace proto {  // easy algorithmic description

template <
    typename T, typename Eq = int32_t, typename Fp = T,
    typename Compact = typename CompactDram<PROPER_GPU_BACKEND, T>::Compact,
    int BLK = 256>
__global__ void c_lorenzo_1d1l(
    T* in_data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq,
    Compact compact)
{
  SETUP_ND_GPU_CUDA;
  __shared__ T buf[BLK];

  auto id = gid1();
  auto data = [&](auto dx) -> T& { return buf[t().x + dx]; };

  // prequant (fp presence)
  if (id < len3.x) { data(0) = round(in_data[id] * ebx2_r); }
  __syncthreads();

  T delta = data(0) - (t().x == 0 ? 0 : data(-1));
  bool quantizable = fabs(delta) < radius;
  T candidate = delta + radius;
  if (check_boundary1()) {  // postquant
    eq[id] = quantizable * static_cast<Eq>(candidate);
    if (not quantizable) {
      auto dram_idx = atomicAdd(compact.d_num, 1);
      compact.d_val[dram_idx] = candidate;
      compact.d_idx[dram_idx] = id;
    }
  }
}

template <
    typename T, typename Eq = int32_t, typename Fp = T,
    typename Compact = typename CompactDram<PROPER_GPU_BACKEND, T>::Compact,
    int BLK = 16>
__global__ void c_lorenzo_2d1l(
    T* in_data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq,
    Compact compact)
{
  SETUP_ND_GPU_CUDA;

  __shared__ T buf[BLK][BLK + 1];

  uint32_t y = threadIdx.y, x = threadIdx.x;
  auto data = [&](auto dx, auto dy) -> T& {
    return buf[t().y + dy][t().x + dx];
  };

  auto id = gid2();

  if (check_boundary2()) { data(0, 0) = round(in_data[id] * ebx2_r); }
  __syncthreads();

  T delta = data(0, 0) - ((x > 0 ? data(-1, 0) : 0) +             // dist=1
                          (y > 0 ? data(0, -1) : 0) -             // dist=1
                          (x > 0 and y > 0 ? data(-1, -1) : 0));  // dist=2

  bool quantizable = fabs(delta) < radius;
  T candidate = delta + radius;
  if (check_boundary2()) {
    eq[id] = quantizable * static_cast<Eq>(candidate);
    if (not quantizable) {
      auto dram_idx = atomicAdd(compact.d_num, 1);
      compact.d_val[dram_idx] = candidate;
      compact.d_idx[dram_idx] = id;
    }
  }
}

template <
    typename T, typename Eq = int32_t, typename Fp = T,
    typename Compact = typename CompactDram<PROPER_GPU_BACKEND, T>::Compact,
    int BLK = 8>
__global__ void c_lorenzo_3d1l(
    T* in_data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq,
    Compact compact)
{
  SETUP_ND_GPU_CUDA;
  __shared__ T buf[BLK][BLK][BLK + 1];

  auto z = t().z, y = t().y, x = t().x;
  auto data = [&](auto dx, auto dy, auto dz) -> T& {
    return buf[t().z + dz][t().y + dy][t().x + dx];
  };

  auto id = gid3();
  if (check_boundary3()) { data(0, 0, 0) = round(in_data[id] * ebx2_r); }
  __syncthreads();

  T delta = data(0, 0, 0) -
            ((z > 0 and y > 0 and x > 0 ? data(-1, -1, -1) : 0)  // dist=3
             - (y > 0 and x > 0 ? data(-1, -1, 0) : 0)           // dist=2
             - (z > 0 and x > 0 ? data(-1, 0, -1) : 0)           //
             - (z > 0 and y > 0 ? data(0, -1, -1) : 0)           //
             + (x > 0 ? data(-1, 0, 0) : 0)                      // dist=1
             + (y > 0 ? data(0, -1, 0) : 0)                      //
             + (z > 0 ? data(0, 0, -1) : 0));                    //

  bool quantizable = fabs(delta) < radius;
  T candidate = delta + radius;
  if (check_boundary3()) {
    eq[id] = quantizable * static_cast<Eq>(candidate);
    if (not quantizable) {
      auto dram_idx = atomicAdd(compact.d_num, 1);
      compact.d_val[dram_idx] = candidate;
      compact.d_idx[dram_idx] = id;
    }
  }
}

template <typename T, typename Eq = int32_t, typename Fp = T, int BLK = 256>
__global__ void x_lorenzo_1d1l(
    Eq* eq, T* scattered_outlier, dim3 len3, dim3 stride3, int radius, Fp ebx2,
    T* xdata)
{
  SETUP_ND_GPU_CUDA;
  __shared__ T buf[BLK];

  auto id = gid1();
  auto data = [&](auto dx) -> T& { return buf[t().x + dx]; };

  if (id < len3.x)
    data(0) = scattered_outlier[id] + static_cast<T>(eq[id]) - radius;  // fuse
  else
    data(0) = 0;
  __syncthreads();

  for (auto d = 1; d < BLK; d *= 2) {
    T n = 0;
    if (t().x >= d)
      n = data(-d);  // like __shfl_up_sync(0x1f, var, d); warp_sync
    __syncthreads();
    if (t().x >= d) data(0) += n;
    __syncthreads();
  }

  if (id < len3.x) { xdata[id] = data(0) * ebx2; }
}

template <typename T, typename Eq = int32_t, typename Fp = T, int BLK = 16>
__global__ void x_lorenzo_2d1l(
    Eq* eq, T* scattered_outlier, dim3 len3, dim3 stride3, int radius, Fp ebx2,
    T* xdata)
{
  SETUP_ND_GPU_CUDA;
  __shared__ T buf[BLK][BLK + 1];

  auto id = gid2();
  auto data = [&](auto dx, auto dy) -> T& {
    return buf[t().y + dy][t().x + dx];
  };

  if (check_boundary2())
    data(0, 0) =
        scattered_outlier[id] + static_cast<T>(eq[id]) - radius;  // fuse
  else
    data(0, 0) = 0;
  __syncthreads();

  for (auto d = 1; d < BLK; d *= 2) {
    T n = 0;
    if (t().x >= d) n = data(-d, 0);
    __syncthreads();
    if (t().x >= d) data(0, 0) += n;
    __syncthreads();
  }

  for (auto d = 1; d < BLK; d *= 2) {
    T n = 0;
    if (t().y >= d) n = data(0, -d);
    __syncthreads();
    if (t().y >= d) data(0, 0) += n;
    __syncthreads();
  }

  if (check_boundary2()) { xdata[id] = data(0, 0) * ebx2; }
}

template <typename T, typename Eq = int32_t, typename Fp = T, int BLK = 8>
__global__ void x_lorenzo_3d1l(
    Eq* eq, T* scattered_outlier, dim3 len3, dim3 stride3, int radius, Fp ebx2,
    T* xdata)
{
  SETUP_ND_GPU_CUDA;
  __shared__ T buf[BLK][BLK][BLK + 1];

  auto id = gid3();
  auto data = [&](auto dx, auto dy, auto dz) -> T& {
    return buf[t().z + dz][t().y + dy][t().x + dx];
  };

  if (check_boundary3())
    data(0, 0, 0) = scattered_outlier[id] + static_cast<T>(eq[id]) - radius;
  else
    data(0, 0, 0) = 0;
  __syncthreads();

  for (auto dist = 1; dist < BLK; dist *= 2) {
    T addend = 0;
    if (t().x >= dist) addend = data(-dist, 0, 0);
    __syncthreads();
    if (t().x >= dist) data(0, 0, 0) += addend;
    __syncthreads();
  }

  for (auto dist = 1; dist < BLK; dist *= 2) {
    T addend = 0;
    if (t().y >= dist) addend = data(0, -dist, 0);
    __syncthreads();
    if (t().y >= dist) data(0, 0, 0) += addend;
    __syncthreads();
  }

  for (auto dist = 1; dist < BLK; dist *= 2) {
    T addend = 0;
    if (t().z >= dist) addend = data(0, 0, -dist);
    __syncthreads();
    if (t().z >= dist) data(0, 0, 0) += addend;
    __syncthreads();
  }

  if (check_boundary3()) { xdata[id] = data(0, 0, 0) * ebx2; }
}

}  // namespace proto
}  // namespace __kernel
}  // namespace cuda_hip
}  // namespace psz

#include "mem/compact.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

template <typename T, typename Eq>
pszerror psz_comp_lproto(
    T* const data, dim3 const len3, double const eb, int const radius,
    Eq* const eq, void* _outlier, float* time_elapsed, void* stream)
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

  using Compact = typename CompactDram<PROPER_GPU_BACKEND, T>::Compact;

  auto outlier = (Compact*)_outlier;

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
    c_lorenzo_1d1l<T, Eq><<<Grid1D, Block1D, 0, (GpuStreamT)stream>>>(
        data, len3, leap3, radius, ebx2_r, eq, *outlier);
  }
  else if (ndim() == 2) {
    c_lorenzo_2d1l<T, Eq><<<Grid2D, Block2D, 0, (GpuStreamT)stream>>>(
        data, len3, leap3, radius, ebx2_r, eq, *outlier);
  }
  else if (ndim() == 3) {
    c_lorenzo_3d1l<T, Eq><<<Grid3D, Block3D, 0, (GpuStreamT)stream>>>(
        data, len3, leap3, radius, ebx2_r, eq, *outlier);
  }
  else {
    throw std::runtime_error("Lorenzo only works for 123-D.");
  }

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(GpuStreamSync(stream));

  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}

template <typename T, typename Eq>
pszerror psz_decomp_lproto(
    Eq* eq, dim3 const len3, T* scattered_outlier, double const eb,
    int const radius, T* xdata, float* time_elapsed, void* stream)
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
    x_lorenzo_1d1l<T, Eq><<<Grid1D, Block1D, 0, (GpuStreamT)stream>>>(
        eq, scattered_outlier, len3, leap3, radius, ebx2, xdata);
  }
  else if (ndim() == 2) {
    x_lorenzo_2d1l<T, Eq><<<Grid2D, Block2D, 0, (GpuStreamT)stream>>>(
        eq, scattered_outlier, len3, leap3, radius, ebx2, xdata);
  }
  else if (ndim() == 3) {
    x_lorenzo_3d1l<T, Eq><<<Grid3D, Block3D, 0, (GpuStreamT)stream>>>(
        eq, scattered_outlier, len3, leap3, radius, ebx2, xdata);
  }

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(GpuStreamSync(stream));

  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}

#endif
