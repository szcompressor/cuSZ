/**
 * @file l23_fzgpu.cuhip.inl
 * @author Jiannan Tian
 * @brief Adapted from l21conf for FZ-GPU (HPDC '23) (compression part). This
 * only reflects WIP rather than the final version for FZ-GPU.
 * @version 0.4
 * @date 2022-12-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "port.hh"
#include "subr_legacy.cuhip.inl"

namespace psz {

template <
    typename T, int TileDim, int Seq, typename Eq = uint16_t, typename Fp = T>
__global__ void KERNEL_CUHIP_c_lorenzo_1d1l_FZGPU_delta_only(
    T* const in_data, dim3 const data_len3, dim3 const data_leap3,
    Eq* const out_eq, Fp const ebx2_r)
{
  namespace subr_v0 = psz::cuda_hip;

  constexpr auto NTHREAD = TileDim / Seq;

  __shared__ T scratch[TileDim];  // for in_data and outlier
  __shared__ Eq s_eq[TileDim];

  T prev{0};
  T thp_data[Seq];

  auto id_base = blockIdx.x * TileDim;

  subr_v0::load_prequant_1d<T, Fp, NTHREAD, Seq>(
      in_data, data_len3.x, id_base, scratch, thp_data, prev, ebx2_r);
  subr_v0::predict_quantize__no_outlier_1d<T, Eq, Seq, true>(
      thp_data, s_eq, prev);
  subr_v0::predict_quantize__no_outlier_1d<T, Eq, Seq, false>(thp_data, s_eq);
  subr_v0::write_1d<Eq, T, NTHREAD, Seq, false>(
      s_eq, nullptr, data_len3.x, id_base, out_eq, nullptr);
}

template <typename T, typename Eq, typename Fp>
__global__ void KERNEL_CUHIP_c_lorenzo_2d1l_FZGPU_delta_only(
    T* const in_data, dim3 const data_len3, dim3 const data_leap3,
    Eq* const out_eq, Fp const ebx2_r)
{
  namespace subr_v0 = psz::cuda_hip;

  constexpr auto TileDim = 16;
  constexpr auto YSEQ = 8;

  T center[YSEQ + 1] = {0};

  auto gix = blockIdx.x * TileDim + threadIdx.x;
  auto giy_base = blockIdx.y * TileDim + threadIdx.y * YSEQ;

  subr_v0::load_prequant_2d<T, Fp, YSEQ>(
      in_data, data_len3.x, gix, data_len3.y, giy_base, data_leap3.y, ebx2_r,
      center);
  subr_v0::predict_2d<T, Eq, YSEQ>(center);
  subr_v0::delta_only::quantize_write_2d<T, Eq, YSEQ>(
      center, data_len3.x, gix, data_len3.y, giy_base, data_leap3.y, out_eq);
}

template <typename T, typename Eq, typename Fp>
__global__ void KERNEL_CUHIP_c_lorenzo_3d1l_FZGPU_delta_only(
    T* const in_data, dim3 const data_len3, dim3 const data_leap3,
    Eq* const out_eq, Fp const ebx2_r)
{
  constexpr auto TileDim = 8;
  __shared__ T s[9][33];
  T delta[TileDim + 1] = {0};  // first el = 0

  const auto gix = blockIdx.x * (TileDim * 4) + threadIdx.x;
  const auto giy = blockIdx.y * TileDim + threadIdx.y;
  const auto giz_base = blockIdx.z * TileDim;
  const auto base_id = gix + giy * data_leap3.y + giz_base * data_leap3.z;

  auto giz = [&](auto z) { return giz_base + z; };
  auto gid = [&](auto z) { return base_id + z * data_leap3.z; };

  auto load_prequant_3d = [&]() {
    if (gix < data_len3.x and giy < data_len3.y) {
      for (auto z = 0; z < TileDim; z++)
        if (giz(z) < data_len3.z)
          delta[z + 1] =
              round(in_data[gid(z)] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();
  };

  auto quantize_write = [&](T delta, auto x, auto y, auto z, auto gid) {
    if (x < data_len3.x and y < data_len3.y and z < data_len3.z)
      out_eq[gid] = static_cast<Eq>(delta);
  };

  ////////////////////////////////////////////////////////////////////////////

  load_prequant_3d();

  for (auto z = TileDim; z > 0; z--) {
    // z-direction
    delta[z] -= delta[z - 1];

    // x-direction
    auto prev_x = __shfl_up_sync(0xffffffff, delta[z], 1, 8);
    if (threadIdx.x % TileDim > 0) delta[z] -= prev_x;

    // y-direction, exchange via shmem
    // ghost padding along y
    s[threadIdx.y + 1][threadIdx.x] = delta[z];
    __syncthreads();

    delta[z] -= (threadIdx.y > 0) * s[threadIdx.y][threadIdx.x];

    // now delta[z] is delta
    quantize_write(delta[z], gix, giy, giz(z - 1), gid(z - 1));
    __syncthreads();
  }
}

}  // namespace psz
