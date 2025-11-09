/**
 * @file spline3.cu
 * @author Jinyang Liu, Shixun Wu, Jiannan Tian
 * @brief A high-level Spline3D wrapper. Allocations are explicitly out of
 * called functions.
 * @version 0.3
 * @date 2021-06-15
 *
 * (copyright to be updated)
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "detail/spline3.inl"
#include "kernel/predictor.hh"
#include "mem/cxx_sp_gpu.h"

constexpr int DEFAULT_BLOCK_SIZE = 384;
constexpr auto BLK = 8;

#define SETUP                                                                                \
  auto div3 = [](dim3 len, dim3 sublen) {                                                    \
    return dim3(                                                                             \
        (len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1, (len.z - 1) / sublen.z + 1); \
  };                                                                                         \
  auto ndim = [&]() {                                                                        \
    if (len3.z == 1 and len3.y == 1)                                                         \
      return 1;                                                                              \
    else if (len3.z == 1 and len3.y != 1)                                                    \
      return 2;                                                                              \
    else                                                                                     \
      return 3;                                                                              \
  };

template <typename T, typename E, typename FP>
int psz::module::GPU_predict_spline(
    T* in_data, stdlen3 const data_len3,       //
    E* out_ectrl, stdlen3 const ectrl_len3,    //
    T* out_anchor, stdlen3 const anchor_len3,  //
    void* _outlier, f8 const ebx2, f8 const eb_r, uint32_t radius, void* stream)
{
#define l3 data_len3

  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };
  auto grid_dim = dim3(div(l3[0], BLK * 4), div(l3[1], BLK), div(l3[2], BLK));

  using Compact2 = _portable::compact_GPU_DRAM2<T, u4>;
  using Compact2_Validx = _portable::compact_cell<T, u4>;
  auto ot = (Compact2*)_outlier;

  cusz::c_spline3d_infprecis_32x8x8data<T*, E*, float, DEFAULT_BLOCK_SIZE, Compact2_Validx*>  //
      <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (GPU_BACKEND_SPECIFIC_STREAM)stream>>>(
          in_data, STDLEN3_TO_DIM3(data_len3), STDLEN3_TO_STRIDE3(data_len3),      //
          out_ectrl, STDLEN3_TO_DIM3(ectrl_len3), STDLEN3_TO_STRIDE3(ectrl_len3),  //
          out_anchor, STDLEN3_TO_STRIDE3(anchor_len3),                             //
          ot->val_idx_d(), ot->num_d(), eb_r, ebx2, radius);

  return 0;

#undef l3
}

template <typename T, typename E, typename FP>
int psz::module::GPU_reverse_predict_spline(
    E* in_ectrl, stdlen3 const ectrl_len3,   //
    T* in_anchor, stdlen3 anchor_len3,       //
    T* out_xdata, stdlen3 const xdata_len3,  //
    f8 const ebx2, f8 const eb_r, uint32_t radius, void* stream)
{
#define l3 xdata_len3

  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };
  auto grid_dim = dim3(div(l3[0], BLK * 4), div(l3[1], BLK), div(l3[2], BLK));

  cusz::x_spline3d_infprecis_32x8x8data<E*, T*, float, DEFAULT_BLOCK_SIZE>  //
      <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0,
         (GPU_BACKEND_SPECIFIC_STREAM)stream>>>                                   //
      (in_ectrl, STDLEN3_TO_DIM3(ectrl_len3), STDLEN3_TO_STRIDE3(ectrl_len3),     //
       in_anchor, STDLEN3_TO_DIM3(anchor_len3), STDLEN3_TO_STRIDE3(anchor_len3),  //
       out_xdata, STDLEN3_TO_DIM3(xdata_len3), STDLEN3_TO_STRIDE3(xdata_len3),    //
       eb_r, ebx2, radius);

  return 0;

#undef l3
}

#define INSTANTIATE_PSZCXX_MODULE_SPLINE__2params(T, E)                                       \
  template int psz::module::GPU_predict_spline<T, E>(                                         \
      T * in_data, stdlen3 const data_len3, E* out_ectrl, stdlen3 const ectrl_len3,           \
      T* out_anchor, stdlen3 const anchor_len3, void* _outlier, f8 const ebx2, f8 const eb_r, \
      uint32_t radius, void* stream);                                                         \
  template int psz::module::GPU_reverse_predict_spline<T, E>(                                 \
      E * in_ectrl, stdlen3 const ectrl_len3, T* in_anchor, stdlen3 const anchor_len3,        \
      T* out_xdata, stdlen3 const xdata_len3, f8 const ebx2, f8 const eb_r, uint32_t radius,  \
      void* stream);

#define INSTANTIATE_PSZCXX_MODULE_SPLINE__1param(T) \
  INSTANTIATE_PSZCXX_MODULE_SPLINE__2params(T, u1); \
  INSTANTIATE_PSZCXX_MODULE_SPLINE__2params(T, u2);

INSTANTIATE_PSZCXX_MODULE_SPLINE__1param(f4);
INSTANTIATE_PSZCXX_MODULE_SPLINE__1param(f8);

#undef SETUP
#undef INSTANTIATE_PSZCXX_MODULE_SPLINE__1param
#undef INSTANTIATE_PSZCXX_MODULE_SPLINE__2params
