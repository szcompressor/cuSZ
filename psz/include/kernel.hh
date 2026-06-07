#ifndef PSZ_KERNEL_HH
#define PSZ_KERNEL_HH

#include <cstddef>
#include <cstdint>

#include "cusz/type.h"
#include "mem/sp_interface.h"

enum class SplineVariant { y24, y25 };

psz_len psz_div3(psz_len len, psz_len sublen);

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

  static int kernel(
      T* const in_data, psz_len const len, Eq* const out_eq, void* out_outlier, u4* out_top1,
      f8 const eb, u2 const radius, void* stream);
  static int compressor_kernel(
      Buf* buf, T* const in_data, psz_len const len, f8 const eb, u2 const radius, void* stream);
};

template <class Types, class Features>
struct GPU_x_lorenzo_nd {
  using T = typename Types::T;
  using Eq = typename Types::Eq;
  using Buf = typename Types::Buf_Comp;

  static int kernel(
      Eq* const in_eq, T* const in_outlier, T* const out_data, psz_len const len, f8 const eb,
      u2 const radius, void* stream);
};

template <typename T, typename Eq>
struct GPU_PROTO_c_lorenzo_nd_with_outlier {
  static int kernel(
      T* const in_data, psz_len const len, Eq* const out_eq, void* out_outlier, f8 const ebx2,
      f8 const ebx2_r, u2 const radius, void* stream);
};

template <typename T, typename Eq>
struct GPU_PROTO_x_lorenzo_nd {
  static int kernel(
      Eq* in_eq, T* in_outlier, T* out_data, psz_len const len, f8 const ebx2, f8 const ebx2_r,
      int const radius, void* stream);
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

template <typename T, typename E, typename Fp = T>
struct GPU_c_spline_y25 {
  static int null() { return PSZ_ABORT_NO_SUCH_PREDICTOR; }
  static int kernel_v1(
      T* data, psz_len const data_len3, T* anchor, psz_len const anchor_len3, E* ectrl,
      void* _outlier, double eb, double rel_eb, uint32_t radius, INTERP_PARAMS& intp_param,
      T* d_profiled_errors, T* h_profiled_errors, u4 const pe_len, void* stream);
};

template <typename T, typename E, typename Fp = T>
struct GPU_c_spline_y24 {
  static int null() { return PSZ_ABORT_NO_SUCH_PREDICTOR; }
  static int kernel_v1(
      T* data, psz_len const data_len3, T* anchor, psz_len const anchor_len3, E* ectrl,
      void* _outlier, double eb, double rel_eb, uint32_t radius, INTERP_PARAMS& intp_param,
      T* d_profiled_errors, T* h_profiled_errors, u4 const pe_len, void* stream);
};

template <typename T, typename E, typename Fp = T>
struct GPU_c_spline {
  static int kernel_v1(
      T* data, psz_len const data_len3, T* anchor, psz_len const anchor_len3, E* ectrl,
      void* _outlier, double eb, double rel_eb, uint32_t radius, INTERP_PARAMS& intp_param,
      T* d_profiled_errors, T* h_profiled_errors, u4 const pe_len, void* stream,
      SplineVariant variant = SplineVariant::y25)
  {
    if (variant == SplineVariant::y24)
      return GPU_c_spline_y24<T, E, Fp>::kernel_v1(
          data, data_len3, anchor, anchor_len3, ectrl, _outlier, eb, rel_eb, radius, intp_param,
          d_profiled_errors, h_profiled_errors, pe_len, stream);
    return GPU_c_spline_y25<T, E, Fp>::kernel_v1(
        data, data_len3, anchor, anchor_len3, ectrl, _outlier, eb, rel_eb, radius, intp_param,
        d_profiled_errors, h_profiled_errors, pe_len, stream);
  }
};

template <typename T, typename E, typename Fp = T>
struct GPU_x_spline_y25 {
  static int null() { return PSZ_ABORT_NO_SUCH_PREDICTOR; }
  static int kernel_v1(
      T* anchor, psz_len const anchor_len3, E* ectrl, T* xdata, psz_len const xdata_len3,
      T* outlier_tmp, double eb, uint32_t radius, INTERP_PARAMS intp_param, void* stream);
};

template <typename T, typename E, typename Fp = T>
struct GPU_x_spline_y24 {
  static int null() { return PSZ_ABORT_NO_SUCH_PREDICTOR; }
  static int kernel_v1(
      T* anchor, psz_len const anchor_len3, E* ectrl, T* xdata, psz_len const xdata_len3,
      T* outlier_tmp, double eb, uint32_t radius, INTERP_PARAMS intp_param, void* stream);
};

template <typename T, typename E, typename Fp = T>
struct GPU_x_spline {
  static int kernel_v1(
      T* anchor, psz_len const anchor_len3, E* ectrl, T* xdata, psz_len const xdata_len3,
      T* outlier_tmp, double eb, uint32_t radius, INTERP_PARAMS intp_param, void* stream,
      SplineVariant variant = SplineVariant::y25)
  {
    if (variant == SplineVariant::y24)
      return GPU_x_spline_y24<T, E, Fp>::kernel_v1(
          anchor, anchor_len3, ectrl, xdata, xdata_len3, outlier_tmp, eb, radius, intp_param,
          stream);
    return GPU_x_spline_y25<T, E, Fp>::kernel_v1(
        anchor, anchor_len3, ectrl, xdata, xdata_len3, outlier_tmp, eb, radius, intp_param,
        stream);
  }
};

template <typename T, typename M>
struct CPU_scatter {
  [[deprecated("To be replaced by kernel_v2.")]] static int kernel(
      T* val, M* idx, int nnz, T* out);

  using ValIdx = _portable::compact_cell<T, M>;
  static int kernel_v2(ValIdx* val_idx, int nnz, T* out);
};

template <typename T, typename M>
struct GPU_scatter {
  static int kernel(T* val, M* idx, int nnz, T* out, void* stream);

  using ValIdx = _portable::compact_cell<T, M>;
  static int kernel_v2(ValIdx* val_idx, int nnz, T* out, void* stream);
};

}  // namespace psz::module

#endif
