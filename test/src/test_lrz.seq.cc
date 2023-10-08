/**
 * @file test_l2_serial.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-02-25
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include "busyheader.hh"
#include "detail/correctness.inl"
#include "kernel/detail/l23.seq.inl"

using T = float;
using FP = float;
using EQ = int32_t;

size_t t1d_len = 256;
psz_dim3 t1d_len3{256, 1, 1};
psz_dim3 t1d_stride3{1, 1, 1};

size_t t2d_len = 256;
psz_dim3 t2d_len3{16, 16, 1};
psz_dim3 t2d_stride3{1, 16, 1};

size_t t3d_len = 512;
psz_dim3 t3d_len3{8, 8, 8};
psz_dim3 t3d_stride3{1, 8, 64};

template <typename FUNC>
bool test1(
    FUNC func, T const* input, size_t len, psz_dim3 len3, psz_dim3 stride3,
    T const* expected_output, std::string funcname)
{
  auto outlier = new CompactSerial<T>;
  outlier->reserve_space(len / 10).malloc();

  auto eq = new EQ[len];
  memset(eq, 0, sizeof(EQ) * len);
  auto radius = 512;

  func(const_cast<T*>(input), len3, stride3, radius, 1, eq, outlier);

  bool ok = true;
  for (auto i = 0; i < len; i++) {
    if (eq[i] != expected_output[i]) {
      ok = false;
      break;
    }
  }
  cout << funcname << " works as expected: " << (ok ? "yes" : "NO") << endl;

  delete[] eq;
  delete outlier;

  return ok;
}

template <typename FUNC>
bool test2(
    FUNC func, EQ const* input, size_t len, psz_dim3 len3, psz_dim3 stride3,
    T const* expected_output, std::string funcname)
{
  auto xdata = new T[len];
  memset(xdata, 0, sizeof(T) * len);
  auto radius = 512;

  func(
      const_cast<EQ*>(input), xdata /* outlier */, len3, stride3, radius, 1,
      xdata);

  bool ok = true;
  for (auto i = 0; i < len; i++) {
    if (xdata[i] != expected_output[i]) {
      ok = false;
      break;
    }
  }
  cout << funcname << " works as expected: " << (ok ? "yes" : "NO") << endl;

  delete[] xdata;

  return ok;
}

template <typename FUNC1, typename FUNC2>
bool test3(
    FUNC1 func1, FUNC2 func2, T const* input, size_t len, psz_dim3 len3,
    psz_dim3 stride3, std::string funcname)
{
  auto outlier = new struct CompactSerial<T>;
  outlier->reserve_space(len / 10).malloc();

  auto eq = new EQ[len];
  memset(eq, 0, sizeof(EQ) * len);
  auto xdata = new T[len];
  memset(xdata, 0, sizeof(T) * len);
  auto radius = 512;

  auto eb = 1e-2;
  auto ebx2 = eb * 2;
  auto ebx2_r = 1 / (eb * 2);

  func1(const_cast<T*>(input), len3, stride3, radius, ebx2_r, eq, outlier);
  {
    // TODO scatter
  }
  func2(eq, xdata /* outlier */, len3, stride3, radius, ebx2, xdata);

  bool ok = true;
  for (auto i = 0; i < len; i++) {
    if (xdata[i] != input[i]) {
      ok = false;
      break;
    }
  }
  cout << funcname << " works as expected: " << (ok ? "yes" : "NO") << endl;

  delete[] eq;
  delete[] xdata;
  delete outlier;

  return ok;
}

template <typename T>
struct FunctionType {
  using FP = T;
  using EQ = int32_t;
  using OUTLIER = CompactSerial<T>;
  typedef std::function<void(T*, psz_dim3, psz_dim3, int, FP, EQ*, OUTLIER*)>
      type_c;
  typedef std::function<void(EQ*, T*, psz_dim3, psz_dim3, int, FP, T*)> type_x;
};

int main()
{
  FunctionType<T>::type_c cl1d1l = psz::seq::__kernel::c_lorenzo_1d1l<T>;
  FunctionType<T>::type_c cl2d1l = psz::seq::__kernel::c_lorenzo_2d1l<T>;
  FunctionType<T>::type_c cl3d1l = psz::seq::__kernel::c_lorenzo_3d1l<T>;

  FunctionType<T>::type_x xl1d1l = psz::seq::__kernel::x_lorenzo_1d1l<T>;
  FunctionType<T>::type_x xl2d1l = psz::seq::__kernel::x_lorenzo_2d1l<T>;
  FunctionType<T>::type_x xl3d1l = psz::seq::__kernel::x_lorenzo_3d1l<T>;

  auto all_pass = true;

  all_pass = all_pass and test1(
                              cl1d1l, t1d_in, t1d_len, t1d_len3, t1d_stride3,
                              t1d_comp_out, "standalone cl1d1l");
  all_pass = all_pass and test1(
                              cl2d1l, t2d_in, t2d_len, t2d_len3, t2d_stride3,
                              t2d_comp_out, "standalone cl2d1l");
  all_pass = all_pass and test1(
                              cl3d1l, t3d_in, t3d_len, t3d_len3, t3d_stride3,
                              t3d_comp_out, "standalone cl3d1l");

  all_pass = all_pass and test2(
                              xl1d1l, t1d_eq, t1d_len, t1d_len3, t1d_stride3,
                              t1d_decomp_out, "standalone xl1d1l");
  all_pass = all_pass and test2(
                              xl2d1l, t2d_eq, t2d_len, t2d_len3, t2d_stride3,
                              t2d_decomp_out, "standalone xl2d1l");
  all_pass = all_pass and test2(
                              xl3d1l, t3d_eq, t3d_len, t3d_len3, t3d_stride3,
                              t3d_decomp_out, "standalone xl3d1l");

  all_pass = all_pass and test3(
                              cl1d1l, xl1d1l, t1d_in, t1d_len, t1d_len3,
                              t1d_stride3, "lorenzo_1d1l");
  all_pass = all_pass and test3(
                              cl2d1l, xl2d1l, t2d_in, t2d_len, t2d_len3,
                              t2d_stride3, "lorenzo_2d1l");
  all_pass = all_pass and test3(
                              cl3d1l, xl3d1l, t3d_in, t3d_len, t3d_len3,
                              t3d_stride3, "lorenzo_3d1l");

  if (all_pass)
    return 0;
  else
    return -1;
}
