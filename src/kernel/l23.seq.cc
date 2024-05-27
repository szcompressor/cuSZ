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

#include "detail/l23.seq.inl"

#include "cusz/type.h"
#include "kernel/lrz/l23.seq.hh"

template <typename T, typename EQ, typename FP, typename OUTLIER>
pszerror psz_comp_l23_seq(
    T* const data, psz_dim3 const len3, f8 const eb, int const radius,
    EQ* const eq, OUTLIER* outlier, f4* time_elapsed)
{
  auto divide3 = [](psz_dim3 len, psz_dim3 sublen) {
    return psz_dim3{
        (len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1,
        (len.z - 1) / sublen.z + 1};
  };

  auto ndim = [&]() {
    if (len3.z == 1 and len3.y == 1)
      return 1;
    else if (len3.z == 1 and len3.y != 1)
      return 2;
    else
      return 3;
  };

  auto d = ndim();

  // error bound
  auto ebx2 = eb * 2;
  auto ebx2_r = 1 / ebx2;
  auto leap3 = psz_dim3{1, len3.x, len3.x * len3.y};

  if (d == 1) {
    psz::seq::__kernel::c_lorenzo_1d1l<T, EQ, FP, 256>(
        data, len3, leap3, radius, ebx2_r, eq, outlier);
  }
  else if (d == 2) {
    psz::seq::__kernel::c_lorenzo_2d1l<T, EQ, FP, 16>(
        data, len3, leap3, radius, ebx2_r, eq, outlier);
  }
  else if (d == 3) {
    psz::seq::__kernel::c_lorenzo_3d1l<T, EQ, FP, 8>(
        data, len3, leap3, radius, ebx2_r, eq, outlier);
  }

  return CUSZ_SUCCESS;
}

template <typename T, typename EQ, typename FP>
pszerror psz_decomp_l23_seq(
    EQ* eq, psz_dim3 const len3, T* outlier, f8 const eb, int const radius,
    T* xdata, f4* time_elapsed)
{
  auto divide3 = [](psz_dim3 len, psz_dim3 sublen) {
    return psz_dim3{
        (len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1,
        (len.z - 1) / sublen.z + 1};
  };

  auto ndim = [&]() {
    if (len3.z == 1 and len3.y == 1)
      return 1;
    else if (len3.z == 1 and len3.y != 1)
      return 2;
    else
      return 3;
  };

  // error bound
  auto ebx2 = eb * 2;
  auto ebx2_r = 1 / ebx2;
  auto leap3 = psz_dim3{1, len3.x, len3.x * len3.y};

  auto d = ndim();

  if (d == 1) {
    psz::seq::__kernel::x_lorenzo_1d1l<T, EQ, FP, 256>(
        eq, outlier, len3, leap3, radius, ebx2, xdata);
  }
  else if (d == 2) {
    psz::seq::__kernel::x_lorenzo_2d1l<T, EQ, FP, 16>(
        eq, outlier, len3, leap3, radius, ebx2, xdata);
  }
  else if (d == 3) {
    psz::seq::__kernel::x_lorenzo_3d1l<T, EQ, FP, 8>(
        eq, outlier, len3, leap3, radius, ebx2, xdata);
  }

  return CUSZ_SUCCESS;
}

#define CPP_INS(T, EQ, FP)                                      \
  template pszerror psz_comp_l23_seq<T, EQ, FP>(                \
      T* const, psz_dim3 const, f8 const, int const, EQ* const, \
      CompactSerial<T>*, f4*);                                  \
                                                                \
  template pszerror psz_decomp_l23_seq<T, EQ, FP>(              \
      EQ*, psz_dim3 const, T*, f8 const, int const, T*, f4*);

CPP_INS(f4, u1, f4);
CPP_INS(f4, u2, f4);
CPP_INS(f4, u4, f4);
CPP_INS(f4, f4, f4);

CPP_INS(f8, u1, f8);
CPP_INS(f8, u2, f8);
CPP_INS(f8, u4, f8);
CPP_INS(f8, f4, f8);

#undef CPP_INS
