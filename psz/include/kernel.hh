#ifndef PSZ_KERNEL_HH
#define PSZ_KERNEL_HH

#include <cstddef>
#include <cstdint>

#include "cusz/component.hh"
#include "cusz/type.h"
#include "mem/sp_interface.h"
#include "mem/view.hh"

enum class SplineVariant { y24, y25 };

psz_len psz_div3(psz_len len, psz_len sublen);

namespace host = _ptb::host;

namespace psz::module {

template <typename E>
int SEQ_histogram_generic(
    E* in_data, size_t const data_len, uint32_t* out_hist, uint16_t const hist_len,
    float* milliseconds);

template <typename E>
struct GPU_histogram_generic {
  static void init(
      size_t const data_len, uint16_t const hist_len, int& grid_dim, int& block_dim,
      int& shmem_use, int& r_per_block);

  static int kernel(
      E* in_data, size_t const data_len, uint32_t* out_hist, uint16_t const hist_len,
      int const grid_dim, int const block_dim, int const shmem_use, int const r_per_block,
      void* stream);
};

template <typename E>
int SEQ_histogram_Cauchy_v2(
    E* in_data, size_t const data_len, uint32_t* out_hist, uint16_t const hist_len,
    float* milliseconds);

template <typename E>
struct GPU_histogram_Cauchy {
  static int kernel(
      E* in_data, size_t const data_len, uint32_t* out_hist, uint16_t const hist_len,
      void* stream);
};

// Lorenzo predictors //////////////////////////////////////////////////////////

template <class Types, class Features>
struct GPU_c_lorenzo_nd {
  using T = typename Types::T;
  using Eq = typename Types::Eq;
  using Buf = typename Types::Buf_Comp;

  static int kernel(Buf* buf, host::view<T> in_data, f8 const eb, u2 const radius, void* stream);
};

template <class Types, class Features>
struct GPU_x_lorenzo_nd {
  using T = typename Types::T;
  using Eq = typename Types::Eq;
  using Buf = typename Types::Buf_Comp;

  static int kernel(Buf* buf, T* out, f8 const eb, u2 const radius, void* stream);
};

template <typename T, bool UseZigZag, typename Eq>
struct CPU_c_lorenzo_nd_with_outlier {
  static int kernel(
      T* const in_data, psz_len const data_len3, Eq* const out_eq, void* out_outlier, f8 const eb,
      u2 const radius, float* time_elapsed);
};

template <typename T, bool UseZigZag, typename Eq>
struct CPU_x_lorenzo_nd {
  static int kernel(
      Eq* const in_eq, T* const in_outlier, T* const out_data, psz_len const data_len3,
      f8 const eb, u2 const radius, f4* time_elapsed);
};

// spline-based interpolation //////////////////////////////////////////////////
// y24: 3D (x32-y8-z8); y25: 2D (x64-y64) and 3D (x16-y16-z16) /////////////////

template <class Types>
struct GPU_c_spline_y24 {
  using T = typename Types::T;
  using E = typename Types::Eq;
  using Buf = typename Types::Buf_Comp;

  static int kernel(
      Buf* buf, host::view<T> data, double eb, double rel_eb, uint32_t radius,
      INTERP_PARAMS& intp_param, void* stream);
};

template <class Types>
struct GPU_c_spline_y25 {
  using T = typename Types::T;
  using E = typename Types::Eq;
  using Buf = typename Types::Buf_Comp;

  static int kernel(
      Buf* buf, host::view<T> data, double eb, double rel_eb, uint32_t radius,
      INTERP_PARAMS& intp_param, void* stream);
};

template <class Types>
struct GPU_x_spline_y24 {
  using T = typename Types::T;
  using E = typename Types::Eq;
  using Buf = typename Types::Buf_Comp;

  static int kernel(
      Buf* buf, T* anchor, host::view<T> xdata, double eb, uint32_t radius,
      INTERP_PARAMS intp_param, void* stream);
};

template <class Types>
struct GPU_x_spline_y25 {
  using T = typename Types::T;
  using E = typename Types::Eq;
  using Buf = typename Types::Buf_Comp;

  static int kernel(
      Buf* buf, T* anchor, host::view<T> xdata, double eb, uint32_t radius,
      INTERP_PARAMS intp_param, void* stream);
};

template <typename T, typename M>
struct CPU_scatter {
  [[deprecated("To be replaced by kernel_v2.")]] static int kernel(
      T* val, M* idx, int nnz, T* out);

  using ValIdx = _ptb::compact_cell<T, M>;
  static int kernel_v2(ValIdx* val_idx, int nnz, T* out);
};

template <typename T, typename M>
struct GPU_scatter {
  using ValIdx = _ptb::compact_cell<T, M>;
  // fuse the compact outliers onto the decoded eq
  static int kernel_v3_fuse(ValIdx* val_idx, int nnz, T* out, void* stream);
};

}  // namespace psz::module

#endif
