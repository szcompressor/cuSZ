/**
 * @file spline3.cu
 * @author Jiannan Tian
 * @brief A high-level Spline3D wrapper. Allocations are explicitly out of
 * called functions.
 * @version 0.3
 * @date 2021-06-15
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "busyheader.hh"
#include "cusz/type.h"
#include "detail/spline3.inl"
#include "kernel/spline.hh"
#include "mem/compact.hh"

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

template <typename T, typename E, typename FP, bool NO_R_SEPARATE>
void spline3_construct_raw(
    T* data, dim3 const len3, T* anchor, dim3 const an_len3, E* ectrl,
    dim3 const ec_len3, double const eb, int const radius, float& time_elapsed,
    void* stream)
{
  SETUP;

  constexpr auto SUBLEN_3D = dim3(32, 8, 8);
  constexpr auto SEQ_3D = dim3(1, 8, 1);
  constexpr auto BLOCK_3D = dim3(256, 1, 1);
  auto GRID_3D = div3(len3, SUBLEN_3D);

  ////////////////////////////////////////

  auto ebx2 = eb * 2;
  auto eb_r = 1 / eb;
  auto leap3 = dim3(1, len3.x, len3.x * len3.y);
  auto ec_leap3 = dim3(1, ec_len3.x, ec_len3.x * ec_len3.y);
  auto an_leap3 = dim3(1, an_len3.x, an_len3.x * an_len3.y);

  auto d = ndim();

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  cusz::c_spline3d_infprecis_32x8x8data<T*, E*, float, 256>  //
      <<<GRID_3D, BLOCK_3D, 0, (GpuStreamT)stream>>>         //
      (data, len3, leap3,                                    //
       ectrl, ec_len3, ec_leap3,                             //
       anchor, an_leap3,                                     //
       eb_r, ebx2, radius);

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(GpuStreamSync(stream));

  TIME_ELAPSED_GPUEVENT(&time_elapsed);
  DESTROY_GPUEVENT_PAIR;
}

template <typename T, typename E, typename FP>
void spline3_reconstruct_raw(
    T* xdata, dim3 const len3, T* anchor, dim3 const an_len3, E* ectrl,
    dim3 const ec_len3, double const eb, int const radius, float& time_elapsed,
    void* stream)
{
  SETUP;

  constexpr auto SUBLEN_3D = dim3(32, 8, 8);
  constexpr auto SEQ_3D = dim3(1, 8, 1);
  constexpr auto BLOCK_3D = dim3(256, 1, 1);
  auto GRID_3D = div3(len3, SUBLEN_3D);

  ////////////////////////////////////////

  auto ebx2 = eb * 2;
  auto eb_r = 1 / eb;
  auto leap3 = dim3(1, len3.x, len3.x * len3.y);
  auto ec_leap3 = dim3(1, ec_len3.x, ec_len3.x * ec_len3.y);
  auto an_leap3 = dim3(1, an_len3.x, an_len3.x * an_len3.y);

  auto d = ndim();

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  cusz::x_spline3d_infprecis_32x8x8data<E*, T*, float, 256>  //
      <<<GRID_3D, BLOCK_3D, 0, (GpuStreamT)stream>>>         //
      (ectrl, ec_len3, ec_leap3,                             //
       anchor, an_len3, an_leap3,                            //
       xdata, len3, leap3,                                   //
       eb_r, ebx2, radius);

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(GpuStreamSync(stream));

  TIME_ELAPSED_GPUEVENT(&time_elapsed);
  DESTROY_GPUEVENT_PAIR;
}

template <typename T, typename E, typename FP>
int spline_construct(
    pszmem_cxx<T>* data, pszmem_cxx<T>* anchor, pszmem_cxx<E>* ectrl,
    void* _outlier, double eb, uint32_t radius, float* time, void* stream)
{
  constexpr auto BLOCK = 8;

  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  auto ebx2 = eb * 2;
  auto eb_r = 1 / eb;

  auto l3 = data->template len3<dim3>();
  auto grid_dim =
      dim3(div(l3.x, BLOCK * 4), div(l3.y, BLOCK), div(l3.z, BLOCK));

  using Compact = typename CompactDram<PROPER_GPU_BACKEND, T>::Compact;
  auto ot = (Compact*)_outlier;

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  cusz::c_spline3d_infprecis_32x8x8data<T*, E*, float, 256>  //
      <<<grid_dim, dim3(256, 1, 1), 0, (GpuStreamT)stream>>>(
          data->dptr(), data->template len3<dim3>(),
          data->template st3<dim3>(),  //
          ectrl->dptr(), ectrl->template len3<dim3>(),
          ectrl->template st3<dim3>(),  //
          anchor->dptr(), anchor->template st3<dim3>(), eb_r, ebx2, radius);

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(GpuStreamSync(stream));
  TIME_ELAPSED_GPUEVENT(time);
  DESTROY_GPUEVENT_PAIR;

  return 0;
}

template <typename T, typename E, typename FP>
int spline_reconstruct(
    pszmem_cxx<T>* anchor, pszmem_cxx<E>* ectrl, pszmem_cxx<T>* xdata,
    double eb, uint32_t radius, float* time, void* stream)
{
  constexpr auto BLOCK = 8;

  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  auto ebx2 = eb * 2;
  auto eb_r = 1 / eb;

  auto l3 = xdata->template len3<dim3>();
  auto grid_dim =
      dim3(div(l3.x, BLOCK * 4), div(l3.y, BLOCK), div(l3.z, BLOCK));

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  cusz::x_spline3d_infprecis_32x8x8data<E*, T*, float, 256>   //
      <<<grid_dim, dim3(256, 1, 1), 0, (GpuStreamT)stream>>>  //
      (ectrl->dptr(), ectrl->template len3<dim3>(),
       ectrl->template st3<dim3>(),  //
       anchor->dptr(), anchor->template len3<dim3>(),
       anchor->template st3<dim3>(),  //
       xdata->dptr(), xdata->template len3<dim3>(),
       xdata->template st3<dim3>(),  //
       eb_r, ebx2, radius);

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(GpuStreamSync(stream));
  TIME_ELAPSED_GPUEVENT(time);
  DESTROY_GPUEVENT_PAIR;

  return 0;
}

#define INIT(T, E)                                                            \
  template int spline_construct<T, E>(                                        \
      pszmem_cxx<T> * data, pszmem_cxx<T> * anchor, pszmem_cxx<E> * ectrl,    \
      void* _outlier, double eb, uint32_t radius, float* time, void* stream); \
  template int spline_reconstruct<T, E>(                                      \
      pszmem_cxx<T> * anchor, pszmem_cxx<E> * ectrl, pszmem_cxx<T> * xdata,   \
      double eb, uint32_t radius, float* time, void* stream);

INIT(f4, u1)
INIT(f4, u2)
INIT(f4, u4)
INIT(f4, f4)

INIT(f8, u1)
INIT(f8, u2)
INIT(f8, u4)
INIT(f8, f4)

#undef INIT
#undef SETUP
