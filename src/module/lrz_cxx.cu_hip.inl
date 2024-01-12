// deps
#include "cusz/cxx_array.hh"
#include "cusz/type.h"
#include "exception/exception.hh"
#include "kernel/lrz.hh"
#include "mem/compact.hh"
#include "typing.hh"
#include "utils/err.hh"
#include "utils/timer.hh"
// definitions
#include "module/cxx_module.hh"
#include "kernel/detail/l23_x.cu_hip.inl"
#include "kernel/detail/l23r.cu_hip.inl"

namespace _2401 {

template <typename T>
pszerror pszcxx_predict_lorenzo(
    pszarray_cxx<T> in, pszrc2 const rc, pszarray_cxx<u4> out_errquant,
    pszcompact_cxx<T> out_outlier, void* stream)
try {
  static_assert(
      std::is_same<u4, u4>::value or std::is_same<u4, uint16_t>::value or
          std::is_same<u4, uint8_t>::value,
      "u4 must be unsigned integer that is less than or equal to 4 bytes.");

  auto div3 = [](dim3 len, dim3 tile) {
    return dim3(
        (len.x - 1) / tile.x + 1, (len.y - 1) / tile.y + 1,
        (len.z - 1) / tile.z + 1);
  };

  auto len3 = dim3(in.len3.x, in.len3.y, in.len3.z);

  auto ndim = [&]() {
    if (len3.z == 1 and len3.y == 1)
      return 1;
    else if (len3.z == 1 and len3.y != 1)
      return 2;
    else
      return 3;
  };

  constexpr auto Tile1D = 256;
  constexpr auto Seq1D = 4;
  constexpr auto Block1D = 64;
  auto Grid1D = div3(len3, Tile1D);

  constexpr auto Tile2D = dim3(16, 16, 1);
  constexpr auto Block2D = dim3(16, 2, 1);
  auto Grid2D = div3(len3, Tile2D);

  constexpr auto Tile3D = dim3(32, 8, 8);
  constexpr auto Block3D = dim3(32, 8, 1);
  auto Grid3D = div3(len3, Tile3D);

  auto d = ndim();

  // error bound
  auto ebx2 = rc.eb * 2;
  auto ebx2_r = 1 / ebx2;
  auto leap3 = dim3(1, len3.x, len3.x * len3.y);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING((cudaStream_t)stream);

  if (d == 1) {
    psz::rolling::c_lorenzo_1d1l<T, u4, T, Tile1D, Seq1D>
        <<<Grid1D, Block1D, 0, (cudaStream_t)stream>>>(
            in.buf, len3, leap3, rc.radius, ebx2_r, out_errquant.buf,
            out_outlier.val, out_outlier.idx, out_outlier.num);
  }
  else if (d == 2) {
    psz::rolling::c_lorenzo_2d1l<T, u4, T>
        <<<Grid2D, Block2D, 0, (cudaStream_t)stream>>>(
            in.buf, len3, leap3, rc.radius, ebx2_r, out_errquant.buf,
            out_outlier.val, out_outlier.idx, out_outlier.num);
  }
  else if (d == 3) {
    psz::rolling::c_lorenzo_3d1l<T, u4, T>
        <<<Grid3D, Block3D, 0, (cudaStream_t)stream>>>(
            in.buf, len3, leap3, rc.radius, ebx2_r, out_errquant.buf,
            out_outlier.val, out_outlier.idx, out_outlier.num);
  }

  float* time_elapsed;

  STOP_GPUEVENT_RECORDING((cudaStream_t)stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}
NONEXIT_CATCH(psz::exception_placeholder, CUSZ_NOT_IMPLEMENTED)
NONEXIT_CATCH(psz::exception_incorrect_type, CUSZ_FAIL_UNSUPPORTED_DATATYPE)

template <typename T>
pszerror pszcxx_reverse_predict_lorenzo(
    pszarray_cxx<u4> in_errquant, pszarray_cxx<T> in_scattered_outlier,
    pszrc2 const rc, pszarray_cxx<T> out_reconstruct, void* stream)
{
  auto div3 = [](dim3 l, dim3 subl) {
    return dim3(
        (l.x - 1) / subl.x + 1, (l.y - 1) / subl.y + 1,
        (l.z - 1) / subl.z + 1);
  };

  auto len3 = dim3(
      out_reconstruct.len3.x, out_reconstruct.len3.y, out_reconstruct.len3.z);

  auto ndim = [&]() {
    if (len3.z == 1 and len3.y == 1)
      return 1;
    else if (len3.z == 1 and len3.y != 1)
      return 2;
    else
      return 3;
  };

  float* time_elapsed;

  constexpr auto Tile1D = 256;
  constexpr auto Seq1D = 8;  // x-sequentiality == 8
  constexpr auto Block1D = dim3(256 / 8, 1, 1);
  auto Grid1D = div3(len3, Tile1D);

  constexpr auto Tile2D = dim3(16, 16, 1);
  // constexpr auto Seq2D    = dim3(1, 8, 1);  // y-sequentiality == 8
  constexpr auto Block2D = dim3(16, 2, 1);
  auto Grid2D = div3(len3, Tile2D);

  constexpr auto Tile3D = dim3(32, 8, 8);
  // constexpr auto Seq3D    = dim3(1, 8, 1);  // y-sequentiality == 8
  constexpr auto Block3D = dim3(32, 1, 8);
  auto Grid3D = div3(len3, Tile3D);

  // error bound
  auto ebx2 = rc.eb * 2;
  auto ebx2_r = 1 / ebx2;
  auto leap3 = dim3(1, len3.x, len3.x * len3.y);

  auto d = ndim();

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING((cudaStream_t)stream);

  if (d == 1) {
    psz::cuda_hip::__kernel::x_lorenzo_1d1l<T, u4, T, Tile1D, Seq1D>
        <<<Grid1D, Block1D, 0, (cudaStream_t)stream>>>(
            in_errquant.buf, in_scattered_outlier.buf, len3, leap3, rc.radius,
            ebx2, out_reconstruct.buf);
  }
  else if (d == 2) {
    psz::cuda_hip::__kernel::x_lorenzo_2d1l<T, u4, T>
        <<<Grid2D, Block2D, 0, (cudaStream_t)stream>>>(
            in_errquant.buf, in_scattered_outlier.buf, len3, leap3, rc.radius,
            ebx2, out_reconstruct.buf);
  }
  else if (d == 3) {
    psz::cuda_hip::__kernel::x_lorenzo_3d1l<T, u4, T>
        <<<Grid3D, Block3D, 0, (cudaStream_t)stream>>>(
            in_errquant.buf, in_scattered_outlier.buf, len3, leap3, rc.radius,
            ebx2, out_reconstruct.buf);
  }

  STOP_GPUEVENT_RECORDING((cudaStream_t)stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}

}  // namespace _2401
