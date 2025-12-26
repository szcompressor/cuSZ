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
    T* in_data, psz_len const len,             //
    E* out_ectrl, psz_len const ectrl_len3,    //
    T* out_anchor, psz_len const anchor_len3,  //
    void* _outlier, f8 const ebx2, f8 const eb_r, uint32_t radius, void* stream)
{
  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };
  auto grid_dim = dim3(div(len.x, BLK * 4), div(len.y, BLK), div(len.z, BLK));

  using Compact2 = _portable::compact_GPU_DRAM2<T, u4>;
  using Compact2_Validx = _portable::compact_cell<T, u4>;
  auto ot = (Compact2*)_outlier;

  cusz::c_spline3d_infprecis_32x8x8data<T*, E*, float, DEFAULT_BLOCK_SIZE, Compact2_Validx*>  //
      <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (GPU_BACKEND_SPECIFIC_STREAM)stream>>>(
          in_data, LEN_TO_DIM3(len), LEN_TO_STRIDE3(len),                  //
          out_ectrl, LEN_TO_DIM3(ectrl_len3), LEN_TO_STRIDE3(ectrl_len3),  //
          out_anchor, LEN_TO_STRIDE3(anchor_len3),                         //
          ot->val_idx_d(), ot->num_d(), eb_r, ebx2, radius);

  return 0;
}

template <typename T, typename E, typename FP>
int psz::module::GPU_reverse_predict_spline(
    E* in_ectrl, psz_len const ectrl_len3, T* in_anchor, psz_len anchor_len3, T* out_xdata,
    psz_len const len, f8 const ebx2, f8 const eb_r, uint32_t radius, void* stream)
{
  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };
  auto grid_dim = dim3(div(len.x, BLK * 4), div(len.y, BLK), div(len.z, BLK));

  cusz::x_spline3d_infprecis_32x8x8data<E*, T*, float, DEFAULT_BLOCK_SIZE>  //
      <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0,
         (GPU_BACKEND_SPECIFIC_STREAM)stream>>>                           //
      (in_ectrl, LEN_TO_DIM3(ectrl_len3), LEN_TO_STRIDE3(ectrl_len3),     //
       in_anchor, LEN_TO_DIM3(anchor_len3), LEN_TO_STRIDE3(anchor_len3),  //
       out_xdata, LEN_TO_DIM3(len), LEN_TO_STRIDE3(len),                  //
       eb_r, ebx2, radius);

  return 0;
}

#define INSTANTIATE_PSZCXX_MODULE_SPLINE__2params(T, E)                                       \
  template int psz::module::GPU_predict_spline<T, E>(                                         \
      T * in_data, psz_len const data_len3, E* out_ectrl, psz_len const ectrl_len3,           \
      T* out_anchor, psz_len const anchor_len3, void* _outlier, f8 const ebx2, f8 const eb_r, \
      uint32_t radius, void* stream);                                                         \
  template int psz::module::GPU_reverse_predict_spline<T, E>(                                 \
      E * in_ectrl, psz_len const ectrl_len3, T* in_anchor, psz_len const anchor_len3,        \
      T* out_xdata, psz_len const xdata_len3, f8 const ebx2, f8 const eb_r, uint32_t radius,  \
      void* stream);

#define INSTANTIATE_PSZCXX_MODULE_SPLINE__1param(T) \
  INSTANTIATE_PSZCXX_MODULE_SPLINE__2params(T, u1); \
  INSTANTIATE_PSZCXX_MODULE_SPLINE__2params(T, u2);

INSTANTIATE_PSZCXX_MODULE_SPLINE__1param(f4);
INSTANTIATE_PSZCXX_MODULE_SPLINE__1param(f8);

#undef SETUP
#undef INSTANTIATE_PSZCXX_MODULE_SPLINE__1param
#undef INSTANTIATE_PSZCXX_MODULE_SPLINE__2params
