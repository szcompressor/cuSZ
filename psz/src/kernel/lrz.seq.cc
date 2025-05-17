/**
 * @file l23.seq.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-03-16
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "detail/lrz.seq.inl"

#include "kernel/predictor.hh"
#include "mem/cxx_sp_cpu.h"

psz_dim3_seq psz_div3(psz_dim3_seq len, psz_dim3_seq sublen)
{
  return psz_dim3_seq{
      (len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1, (len.z - 1) / sublen.z + 1};
}

int psz_ndim(psz_dim3_seq len3)
{
  if (len3.z == 1 and len3.y == 1)
    return 1;
  else if (len3.z == 1 and len3.y != 1)
    return 2;
  else
    return 3;
}

namespace psz::module {

template <typename T, bool UseZigZag, typename Eq>
pszerror CPU_c_lorenzo_nd_with_outlier(
    T* const in_data, psz_dim3_seq const data_len3, Eq* const out_eq, void* out_outlier,  //
    f8 const eb, uint16_t const radius, f4* time_elapsed)
{
  auto d = psz_ndim(data_len3);
  auto ebx2 = eb * 2, ebx2_r = 1 / (eb * 2);  // error bound
  auto leap3 = psz_dim3_seq{1, data_len3.x, data_len3.x * data_len3.y};

  if (d == 1)
    psz::KERNEL_SEQ_c_lorenzo_1d1l<T, UseZigZag, Eq>(
        in_data, data_len3, leap3, radius, ebx2_r, out_eq, out_outlier);
  else if (d == 2)
    psz::KERNEL_SEQ_c_lorenzo_2d1l<T, UseZigZag, Eq>(
        in_data, data_len3, leap3, radius, ebx2_r, out_eq, out_outlier);
  else if (d == 3)
    psz::KERNEL_SEQ_c_lorenzo_3d1l<T, UseZigZag, Eq>(
        in_data, data_len3, leap3, radius, ebx2_r, out_eq, out_outlier);

  return CUSZ_SUCCESS;
}

template <typename T, bool UseZigZag, typename Eq>
pszerror CPU_x_lorenzo_nd(
    Eq* in_eq, T* in_outlier, T* out_data, psz_dim3_seq const data_len3,  //
    f8 const eb, uint16_t const radius, f4* time_elapsed)
{
  auto d = psz_ndim(data_len3);
  auto ebx2 = eb * 2, ebx2_r = 1 / (eb * 2);  // error bound
  auto leap3 = psz_dim3_seq{1, data_len3.x, data_len3.x * data_len3.y};

  if (d == 1)
    psz::KERNEL_SEQ_x_lorenzo_1d1l<T, UseZigZag, Eq>(
        in_eq, in_outlier, data_len3, leap3, radius, ebx2, out_data);
  else if (d == 2)
    psz::KERNEL_SEQ_x_lorenzo_2d1l<T, UseZigZag, Eq>(
        in_eq, in_outlier, data_len3, leap3, radius, ebx2, out_data);
  else if (d == 3)
    psz::KERNEL_SEQ_x_lorenzo_3d1l<T, UseZigZag, Eq>(
        in_eq, in_outlier, data_len3, leap3, radius, ebx2, out_data);

  return CUSZ_SUCCESS;
}

}  // namespace psz::module

#define CPP_INS(T, UseZigZag, Eq)                                                            \
  template pszerror psz::module::CPU_c_lorenzo_nd_with_outlier<T, UseZigZag, Eq>(            \
      T* const in_data, psz_dim3_seq const data_len3, Eq* const out_eq, void* out_outlier,   \
      double const eb, uint16_t const radius, float* time_elapsed);                          \
  template pszerror psz::module::CPU_x_lorenzo_nd<T, UseZigZag, Eq>(                         \
      Eq* const in_eq, T* const in_outlier, T* const out_data, psz_dim3_seq const data_len3, \
      f8 const eb, uint16_t const radius, f4* time_elapsed);

CPP_INS(f4, true, u1);
CPP_INS(f4, true, u2);
CPP_INS(f4, false, u1);
CPP_INS(f4, false, u2);

CPP_INS(f8, true, u1);
CPP_INS(f8, true, u2);
CPP_INS(f8, false, u1);
CPP_INS(f8, false, u2);

#undef CPP_INS
