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

#include "busyheader.hh"
#include "cusz/type.h"
#include "detail/spline3.inl"
#include "kernel/spline.hh"
#include "mem/compact.hh"

constexpr int DEFAULT_BLOCK_SIZE = 384;

#define SETUP                                                   \
  auto div3 = [](dim3 len, dim3 sublen) {                       \
    return dim3(                                                \
        (len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1, \
        (len.z - 1) / sublen.z + 1);                            \
  };                                                            \
  auto ndim = [&]() {                                           \
    if (len3.z == 1 and len3.y == 1)                            \
      return 1;                                                 \
    else if (len3.z == 1 and len3.y != 1)                       \
      return 2;                                                 \
    else                                                        \
      return 3;                                                 \
  };

template <typename T, typename E, typename FP>
int pszcxx_predict_spline(
    memobj<T>* data, memobj<T>* anchor, memobj<E>* ectrl, void* _outlier,
    double eb, uint32_t radius, float* time, void* stream)
{
  constexpr auto BLOCK = 8;

  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  auto ebx2 = eb * 2;
  auto eb_r = 1 / eb;

  auto l3 = data->len3();
  auto grid_dim =
      dim3(div(l3.x, BLOCK * 4), div(l3.y, BLOCK), div(l3.z, BLOCK));

  using Compact = typename CompactDram<PROPER_GPU_BACKEND, T>::Compact;
  auto ot = (Compact*)_outlier;

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  cusz::c_spline3d_infprecis_32x8x8data<T*, E*, float, DEFAULT_BLOCK_SIZE>  //
      <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
          data->dptr(), data->len3(), data->st3(),     //
          ectrl->dptr(), ectrl->len3(), ectrl->st3(),  //
          anchor->dptr(), anchor->st3(), ot->val(), ot->idx(), ot->num(), eb_r,
          ebx2, radius);

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time);
  DESTROY_GPUEVENT_PAIR;

  return 0;
}

template <typename T, typename E, typename FP>
int psz::cuhip::GPU_predict_spline(
    T* in_data, dim3 const data_len3, dim3 const data_stride3, E* out_ectrl,
    dim3 const ectrl_len3, dim3 const ectrl_stride3, T* out_anchor,
    dim3 const anchor_len3, dim3 const anchor_stride3, void* _outlier,
    double eb, uint32_t radius, float* time, void* stream)
{
#define l3 data_len3

  constexpr auto BLOCK = 8;
  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };
  auto ebx2 = eb * 2, eb_r = 1 / eb;
  auto grid_dim =
      dim3(div(l3.x, BLOCK * 4), div(l3.y, BLOCK), div(l3.z, BLOCK));

  using Compact = typename CompactDram<PROPER_GPU_BACKEND, T>::Compact;
  auto ot = (Compact*)_outlier;

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  cusz::c_spline3d_infprecis_32x8x8data<T*, E*, float, DEFAULT_BLOCK_SIZE>  //
      <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
          in_data, data_len3, data_stride3, out_ectrl, ectrl_len3,
          ectrl_stride3, out_anchor, anchor_stride3, ot->val(), ot->idx(),
          ot->num(), eb_r, ebx2, radius);

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time);
  DESTROY_GPUEVENT_PAIR;

  return 0;

#undef l3
}

template <typename T, typename E, typename FP>
int pszcxx_reverse_predict_spline(
    memobj<T>* anchor, memobj<E>* ectrl, memobj<T>* xdata, double eb,
    uint32_t radius, float* time, void* stream)
{
  constexpr auto BLOCK = 8;

  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  auto ebx2 = eb * 2;
  auto eb_r = 1 / eb;

  auto l3 = xdata->len3();
  auto grid_dim =
      dim3(div(l3.x, BLOCK * 4), div(l3.y, BLOCK), div(l3.z, BLOCK));

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  cusz::x_spline3d_infprecis_32x8x8data<E*, T*, float, DEFAULT_BLOCK_SIZE>  //
      <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0,
         (cudaStream_t)stream>>>                       //
      (ectrl->dptr(), ectrl->len3(), ectrl->st3(),     //
       anchor->dptr(), anchor->len3(), anchor->st3(),  //
       xdata->dptr(), xdata->len3(), xdata->st3(),     //
       eb_r, ebx2, radius);

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time);
  DESTROY_GPUEVENT_PAIR;

  return 0;
}

template <typename T, typename E, typename FP>
int psz::cuhip::GPU_reverse_predict_spline(
    E* in_ectrl, dim3 const ectrl_len3, dim3 const ectrl_stride3, T* in_anchor,
    dim3 const anchor_len3, dim3 const anchor_stride3, T* out_xdata,
    dim3 const xdata_len3, dim3 const xdata_stride3, double eb,
    uint32_t radius, float* time, void* stream)
{
#define l3 xdata_len3

  constexpr auto BLOCK = 8;
  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };
  auto ebx2 = eb * 2, eb_r = 1 / eb;
  auto grid_dim =
      dim3(div(l3.x, BLOCK * 4), div(l3.y, BLOCK), div(l3.z, BLOCK));

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  cusz::x_spline3d_infprecis_32x8x8data<E*, T*, float, DEFAULT_BLOCK_SIZE>  //
      <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0,
         (cudaStream_t)stream>>>                //
      (in_ectrl, ectrl_len3, ectrl_stride3,     //
       in_anchor, anchor_len3, anchor_stride3,  //
       out_xdata, xdata_len3, xdata_stride3,    //
       eb_r, ebx2, radius);

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time);
  DESTROY_GPUEVENT_PAIR;

  return 0;

#undef l3
}

#define INSTANTIATE_PSZCXX_MODULE_SPLINE__2params(T, E)                       \
  template int pszcxx_predict_spline<T, E>(                                   \
      memobj<T> * data, memobj<T> * anchor, memobj<E> * ectrl,                \
      void* _outlier, double eb, uint32_t radius, float* time, void* stream); \
  template int pszcxx_reverse_predict_spline<T, E>(                           \
      memobj<T> * anchor, memobj<E> * ectrl, memobj<T> * xdata, double eb,    \
      uint32_t radius, float* time, void* stream);                            \
  template int psz::cuhip::GPU_predict_spline<T, E>(                          \
      T * in_data, dim3 const data_len3, dim3 const data_stride3,             \
      E* out_ectrl, dim3 const ectrl_len3, dim3 const ectrl_stride3,          \
      T* out_anchor, dim3 const anchor_len3, dim3 const anchor_stride3,       \
      void* _outlier, double eb, uint32_t radius, float* time, void* stream); \
  template int psz::cuhip::GPU_reverse_predict_spline<T, E>(                  \
      E * in_ectrl, dim3 const ectrl_len3, dim3 const ectrl_stride3,          \
      T* in_anchor, dim3 const anchor_len3, dim3 const anchor_stride3,        \
      T* out_xdata, dim3 const xdata_len3, dim3 const xdata_stride3,          \
      double eb, uint32_t radius, float* time, void* stream);

#define INSTANTIATE_PSZCXX_MODULE_SPLINE__1param(T) \
  INSTANTIATE_PSZCXX_MODULE_SPLINE__2params(T, u1); \
  INSTANTIATE_PSZCXX_MODULE_SPLINE__2params(T, u2); \
  INSTANTIATE_PSZCXX_MODULE_SPLINE__2params(T, u4); \
  INSTANTIATE_PSZCXX_MODULE_SPLINE__2params(T, f4);

INSTANTIATE_PSZCXX_MODULE_SPLINE__1param(f4);
INSTANTIATE_PSZCXX_MODULE_SPLINE__1param(f8);

#undef SETUP
#undef INSTANTIATE_PSZCXX_MODULE_SPLINE__1param
#undef INSTANTIATE_PSZCXX_MODULE_SPLINE__2params
