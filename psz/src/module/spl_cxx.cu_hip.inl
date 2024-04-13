#include "cusz/cxx_array.hh"
#include "cusz/type.h"
#include "kernel/detail/spline3.inl"
#include "exception/exception.hh"
#include "kernel/spline.hh"
#include "mem/compact.hh"
#include "typing.hh"

constexpr int DEFAULT_BLOCK_SIZE = 384;

namespace _2401 {

template <typename T>
pszerror pszcxx_predict_spline(
    pszarray_cxx<T> in, pszrc2 const rc, pszarray_cxx<u4> out_errquant,
    pszcompact_cxx<T> out_outlier, pszarray_cxx<T> out_anchor, float* time,
    void* stream)
try {
  constexpr auto BLOCK = 8;
  using FP = T;
  using E = u4;
  auto Div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };
  auto Len3 = [](auto _l3) -> dim3 { return dim3(1, _l3.x, _l3.x * _l3.y); };
  auto Stride3 = [](auto _l3) -> dim3 { return dim3(_l3.x, _l3.y, _l3.z); };
  auto ebx2 = rc.eb * 2, eb_r = 1 / rc.eb;
  auto len3 = Len3(in.len3);
  auto grid_dim =
      dim3(Div(len3.x, BLOCK * 4), Div(len3.y, BLOCK), Div(len3.z, BLOCK));

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  cusz::c_spline3d_infprecis_32x8x8data<T*, u4*, float, DEFAULT_BLOCK_SIZE>
      <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (GpuStreamT)stream>>>(
          in.buf, Len3(in.len3), Stride3(in.len3), out_errquant.buf,
          Len3(in.len3), Stride3(out_errquant.len3), out_anchor.buf,
          Stride3(out_anchor.len3), out_outlier.val, out_outlier.idx,
          out_outlier.num, eb_r, ebx2, rc.radius);

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(GpuStreamSync(stream));
  TIME_ELAPSED_GPUEVENT(time);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}
NONEXIT_CATCH(psz::exception_placeholder, CUSZ_NOT_IMPLEMENTED)
NONEXIT_CATCH(psz::exception_incorrect_type, CUSZ_FAIL_UNSUPPORTED_DATATYPE)

template <typename T>
pszerror pszcxx_reverse_predict_spline(
    pszarray_cxx<u4> in_errquant, pszarray_cxx<T> in_scattered_outlier,
    pszarray_cxx<T> in_anchor, pszrc2 const rc,
    pszarray_cxx<T> out_reconstruct, float* time, void* stream)
try {
  constexpr auto BLOCK = 8;
  using FP = T;
  using E = u4;
  auto Div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };
  auto Len3 = [](auto _l3) -> dim3 { return dim3(1, _l3.x, _l3.x * _l3.y); };
  auto Stride3 = [](auto _l3) -> dim3 { return dim3(_l3.x, _l3.y, _l3.z); };
  auto ebx2 = rc.eb * 2, eb_r = 1 / rc.eb;
  auto len3 = Len3(out_reconstruct.len3);
  auto grid_dim =
      dim3(Div(len3.x, BLOCK * 4), Div(len3.y, BLOCK), Div(len3.z, BLOCK));

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  cusz::x_spline3d_infprecis_32x8x8data<E*, T*, float, DEFAULT_BLOCK_SIZE>
      <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (GpuStreamT)stream>>>(
          in_errquant.buf, Len3(in_errquant.len3), Stride3(in_errquant.len3),
          in_anchor.buf, Len3(in_anchor.len3), Stride3(in_anchor.len3),
          out_reconstruct.buf, Len3(out_reconstruct.len3),
          Stride3(out_reconstruct.len3), eb_r, ebx2, rc.radius);

  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(GpuStreamSync(stream));
  TIME_ELAPSED_GPUEVENT(time);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}
NONEXIT_CATCH(psz::exception_placeholder, CUSZ_NOT_IMPLEMENTED)
NONEXIT_CATCH(psz::exception_incorrect_type, CUSZ_FAIL_UNSUPPORTED_DATATYPE)

}  // namespace _2401