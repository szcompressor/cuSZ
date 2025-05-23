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

#include "kernel/predictor.hh"
#include "mem/cxx_sp_gpu.h"
#include "proto_lrz_helper.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

#define Z(LEN3) LEN3[2]
#define Y(LEN3) LEN3[1]
#define X(LEN3) LEN3[0]
#define TO_DIM3(LEN3) dim3(X(LEN3), Y(LEN3), Z(LEN3))

namespace psz {

// easy algorithmic description

template <
    typename T, int TileDim = 256, typename Eq = uint16_t, typename CompactVal = T,
    typename CompactIdx = uint32_t, typename CompactNum = uint32_t, typename Fp = T>
__global__ void KERNEL_CUHIP_prototype_c_lorenzo_1d1l(
    T* const in_data, dim3 const data_len3, dim3 const data_leap3, Eq* const out_eq,
    CompactVal* const out_cval, CompactIdx* const out_cidx, CompactNum* const out_cn,
    uint16_t const radius, Fp const ebx2_r)
{
  SETUP_ND_GPU_CUDA;
  __shared__ T buf[TileDim];

  auto id = gid1();
  auto data = [&](auto dx) -> T& { return buf[t().x + dx]; };

  // prequant (fp presence)
  if (id < data_len3.x) { data(0) = round(in_data[id] * ebx2_r); }
  __syncthreads();

  T delta = data(0) - (t().x == 0 ? 0 : data(-1));
  bool quantizable = fabs(delta) < radius;
  T candidate = delta + radius;
  if (check_boundary1()) {  // postquant
    out_eq[id] = quantizable * static_cast<Eq>(candidate);
    if (not quantizable) {
      auto cur_idx = atomicAdd(out_cn, 1);
      out_cidx[cur_idx] = id;
      out_cval[cur_idx] = candidate;
    }
  }
}

template <
    typename T, int TileDim = 16, typename Eq = uint16_t, typename CompactVal = T,
    typename CompactIdx = uint32_t, typename CompactNum = uint32_t, typename Fp = T>
__global__ void KERNEL_CUHIP_prototype_c_lorenzo_2d1l(
    T* const in_data, dim3 const data_len3, dim3 const data_leap3, Eq* const out_eq,
    CompactVal* const out_cval, CompactIdx* const out_cidx, CompactNum* const out_cn,
    uint16_t const radius, Fp const ebx2_r)
{
  SETUP_ND_GPU_CUDA;

  __shared__ T buf[TileDim][TileDim + 1];

  uint32_t y = threadIdx.y, x = threadIdx.x;
  auto data = [&](auto dx, auto dy) -> T& { return buf[t().y + dy][t().x + dx]; };

  auto id = gid2();

  if (check_boundary2()) { data(0, 0) = round(in_data[id] * ebx2_r); }
  __syncthreads();

  T delta = data(0, 0) - ((x > 0 ? data(-1, 0) : 0) +             // dist=1
                          (y > 0 ? data(0, -1) : 0) -             // dist=1
                          (x > 0 and y > 0 ? data(-1, -1) : 0));  // dist=2

  bool quantizable = fabs(delta) < radius;
  T candidate = delta + radius;
  if (check_boundary2()) {
    out_eq[id] = quantizable * static_cast<Eq>(candidate);
    if (not quantizable) {
      auto cur_idx = atomicAdd(out_cn, 1);
      out_cidx[cur_idx] = id;
      out_cval[cur_idx] = candidate;
    }
  }
}

template <
    typename T, int TileDim = 8, typename Eq = uint16_t, typename CompactVal = T,
    typename CompactIdx = uint32_t, typename CompactNum = uint32_t, typename Fp = T>
__global__ void KERNEL_CUHIP_prototype_c_lorenzo_3d1l(
    T* const in_data, dim3 const data_len3, dim3 const data_leap3, Eq* const out_eq,
    CompactVal* const out_cval, CompactIdx* const out_cidx, CompactNum* const out_cn,
    uint16_t const radius, Fp const ebx2_r)
{
  SETUP_ND_GPU_CUDA;
  __shared__ T buf[TileDim][TileDim][TileDim + 1];

  auto z = t().z, y = t().y, x = t().x;
  auto data = [&](auto dx, auto dy, auto dz) -> T& {
    return buf[t().z + dz][t().y + dy][t().x + dx];
  };

  auto id = gid3();
  if (check_boundary3()) { data(0, 0, 0) = round(in_data[id] * ebx2_r); }
  __syncthreads();

  T delta = data(0, 0, 0) - ((z > 0 and y > 0 and x > 0 ? data(-1, -1, -1) : 0)  // dist=3
                             - (y > 0 and x > 0 ? data(-1, -1, 0) : 0)           // dist=2
                             - (z > 0 and x > 0 ? data(-1, 0, -1) : 0)           //
                             - (z > 0 and y > 0 ? data(0, -1, -1) : 0)           //
                             + (x > 0 ? data(-1, 0, 0) : 0)                      // dist=1
                             + (y > 0 ? data(0, -1, 0) : 0)                      //
                             + (z > 0 ? data(0, 0, -1) : 0));                    //

  bool quantizable = fabs(delta) < radius;
  T candidate = delta + radius;
  if (check_boundary3()) {
    out_eq[id] = quantizable * static_cast<Eq>(candidate);
    if (not quantizable) {
      auto cur_idx = atomicAdd(out_cn, 1);
      out_cidx[cur_idx] = id;
      out_cval[cur_idx] = candidate;
    }
  }
}

}  // namespace psz

namespace psz::module {

template <typename T, typename Eq>
int GPU_PROTO_c_lorenzo_nd_with_outlier<T, Eq>::kernel(
    T* const in_data, std::array<size_t, 3> const _data_len3, Eq* const out_eq, void* out_outlier,
    f8 const ebx2, f8 const ebx2_r, uint16_t const radius, void* stream)
{
  auto data_len3 = TO_DIM3(_data_len3);

  auto divide3 = [](dim3 len, dim3 sublen) {
    return dim3(
        (len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1, (len.z - 1) / sublen.z + 1);
  };

  auto ndim = [&]() {
    if (data_len3.z == 1 and data_len3.y == 1)
      return 1;
    else if (data_len3.z == 1 and data_len3.y != 1)
      return 2;
    else
      return 3;
  };

  using Compact = _portable::compact_gpu<T>;
  auto ot = (Compact*)out_outlier;

  constexpr auto Tile1D = dim3(256, 1, 1), Tile2D = dim3(16, 16, 1), Tile3D = dim3(8, 8, 8);
  constexpr auto Block1D = dim3(256, 1, 1), Block2D = dim3(16, 16, 1), Block3D = dim3(8, 8, 8);

  auto Grid1D = divide3(data_len3, Tile1D), Grid2D = divide3(data_len3, Tile2D),
       Grid3D = divide3(data_len3, Tile3D);

  // error bound
  auto data_leap3 = dim3(1, data_len3.x, data_len3.x * data_len3.y);

  if (ndim() == 1)
    psz::KERNEL_CUHIP_prototype_c_lorenzo_1d1l<T>
        <<<Grid1D, Block1D, 0, (GPU_BACKEND_SPECIFIC_STREAM)stream>>>(
            in_data, data_len3, data_leap3, out_eq, ot->val(), ot->idx(), ot->num(), radius,
            ebx2_r);
  else if (ndim() == 2)
    psz::KERNEL_CUHIP_prototype_c_lorenzo_2d1l<T>
        <<<Grid2D, Block2D, 0, (GPU_BACKEND_SPECIFIC_STREAM)stream>>>(
            in_data, data_len3, data_leap3, out_eq, ot->val(), ot->idx(), ot->num(), radius,
            ebx2_r);
  else if (ndim() == 3)
    psz::KERNEL_CUHIP_prototype_c_lorenzo_3d1l<T>
        <<<Grid3D, Block3D, 0, (GPU_BACKEND_SPECIFIC_STREAM)stream>>>(
            in_data, data_len3, data_leap3, out_eq, ot->val(), ot->idx(), ot->num(), radius,
            ebx2_r);
  else
    return CUSZ_NOT_IMPLEMENTED;

  return CUSZ_SUCCESS;
}

}  // namespace psz::module
