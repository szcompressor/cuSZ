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

#include "detail/busyheader.hh"
#include "detail/correctness.inl"
#include "kernel/detail/lrz.seq.inl"
#include "kernel/spv.hh"

using T = float;
using FP = float;
using Eq = uint16_t;

static const size_t t1_len = 256;
static const psz_dim3_seq t1_len3{256, 1, 1};
static const psz_dim3_seq t1_leap3{1, 1, 1};

static const size_t t2_len = 256;
static const psz_dim3_seq t2_len3{16, 16, 1};
static const psz_dim3_seq t2_leap3{1, 16, 1};

static const size_t t3_len = 512;
static const psz_dim3_seq t3_len3{8, 8, 8};
static const psz_dim3_seq t3_leap3{1, 8, 64};

static const uint16_t radius = 512;

template <typename FUNC>
bool test1(
    FUNC func, T* input, size_t len, psz_dim3_seq len3, psz_dim3_seq leap3, T const* expected,
    std::string funcname)
{
  auto outlier = new _portable::compact_seq<T>(len / 10);
  outlier->malloc();

  auto eq = new Eq[len];
  memset(eq, 0, sizeof(Eq) * len);

  func(input, len3, leap3, radius, 1, eq, outlier);

  bool ok = true;
  for (auto i = 0; i < len; i++) {
    if (eq[i] != expected[i] + radius) {
      cout << "eq[" << i << "] = " << eq[i] << " != " << expected[i] << endl;
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
    FUNC func, Eq* input_no_offset, size_t len, psz_dim3_seq len3, psz_dim3_seq leap3,
    T const* expected, std::string funcname)
{
  auto xdata = new T[len];
  memset(xdata, 0, sizeof(T) * len);

  auto input = new Eq[len];
  for (auto i = 0; i < len; i++) input[i] = input_no_offset[i] + radius;

  func(input, xdata /* outlier */, len3, leap3, radius, 1, xdata);

  bool ok = true;
  for (auto i = 0; i < len; i++) {
    if (xdata[i] != expected[i]) {
      cout << "xdata[" << i << "] = " << xdata[i] << " != " << expected[i] << endl;
      ok = false;
      break;
    }
  }
  cout << funcname << " works as expected: " << (ok ? "yes" : "NO") << endl;

  delete[] xdata;
  delete[] input;

  return ok;
}

template <typename FUNC1, typename FUNC2>
bool test3(
    FUNC1 func1, FUNC2 func2, T* input, size_t len, psz_dim3_seq len3, psz_dim3_seq leap3,
    std::string funcname)
{
  auto outlier = new _portable::compact_seq<T>(len / 10);
  outlier->malloc();

  auto eq = new Eq[len];
  memset(eq, 0, sizeof(Eq) * len);
  auto xdata = new T[len];
  memset(xdata, 0, sizeof(T) * len);

  // auto eb = 1e-2;
  auto eb = .5;
  auto ebx2 = eb * 2, ebx2_r = 1 / (eb * 2);

  func1(input, len3, leap3, radius, ebx2_r, eq, outlier);

  psz::spv_scatter_naive<SEQ, T, u4>(
      outlier->val(), outlier->idx(), outlier->num(), input, nullptr);

  func2(eq, xdata /* outlier */, len3, leap3, radius, ebx2, xdata);

  bool ok = true;
  for (auto i = 0; i < len; i++) {
    if (xdata[i] != input[i]) {
      cout << "xdata[" << i << "] = " << xdata[i] << " != " << input[i] << endl;
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
struct Func {
  using Eq = uint16_t;
  using type_c = std::function<void(T*, psz_dim3_seq, psz_dim3_seq, uint16_t, f8, Eq*, void*)>;
  using type_x = std::function<void(Eq*, T*, psz_dim3_seq, psz_dim3_seq, uint16_t, f8, T*)>;
};

int main()
{
  Func<T>::type_c LRZ_C1 = psz::KERNEL_SEQ_c_lorenzo_1d1l<T, false>;
  Func<T>::type_c LRZ_C2 = psz::KERNEL_SEQ_c_lorenzo_2d1l<T, false>;
  Func<T>::type_c LRZ_C3 = psz::KERNEL_SEQ_c_lorenzo_3d1l<T, false>;

  Func<T>::type_x LRZ_X1 = psz::KERNEL_SEQ_x_lorenzo_1d1l<T, false>;
  Func<T>::type_x LRZ_X2 = psz::KERNEL_SEQ_x_lorenzo_2d1l<T, false>;
  Func<T>::type_x LRZ_X3 = psz::KERNEL_SEQ_x_lorenzo_3d1l<T, false>;

  auto P = true;

  // clang-format off
  P = P and test1(LRZ_C1, const_cast<T*>(t1_in), t1_len, t1_len3, t1_leap3, t1_comp_out, "LRZ_C1");
  P = P and test1(LRZ_C2, const_cast<T*>(t2_in), t2_len, t2_len3, t2_leap3, t2_comp_out, "LRZ_C2");
  P = P and test1(LRZ_C3, const_cast<T*>(t3_in), t3_len, t3_len3, t3_leap3, t3_comp_out, "LRZ_C3");
  printf("\n");

  P = P and test2(LRZ_X1, const_cast<Eq*>(t1_eq), t1_len, t1_len3, t1_leap3, t1_decomp_out, "LRZ_X1");
  P = P and test2(LRZ_X2, const_cast<Eq*>(t2_eq), t2_len, t2_len3, t2_leap3, t2_decomp_out, "LRZ_X2");
  P = P and test2(LRZ_X3, const_cast<Eq*>(t3_eq), t3_len, t3_len3, t3_leap3, t3_decomp_out, "LRZ_X3");
  printf("\n");

  P = P and test3(LRZ_C1, LRZ_X1, const_cast<T*>(t1_in), t1_len, t1_len3, t1_leap3, "LRZ_CX1");
  P = P and test3(LRZ_C2, LRZ_X2, const_cast<T*>(t2_in), t2_len, t2_len3, t2_leap3, "LRZ_CX2");
  P = P and test3(LRZ_C3, LRZ_X3, const_cast<T*>(t3_in), t3_len, t3_len3, t3_leap3, "LRZ_CX3");
  // clang-format on

  if (P)
    return 0;
  else
    return -1;
}
