#ifndef PSZ_KERNEL_PREDITOR_HH
#define PSZ_KERNEL_PREDITOR_HH

#include <cstdint>

#include "cusz/type.h"

psz_len psz_div3(psz_len len, psz_len sublen);

namespace psz::module {

template <typename T, class PC, class Buf>
struct GPU_c_lorenzo_nd {
  static int kernel(
      T* const in_data, psz_len const len, typename PC::Eq* const out_eq, void* out_outlier,
      u4* out_top1, f8 const eb, u2 const radius, void* stream);
  static int compressor_kernel(
      Buf* buf, T* const in_data, psz_len const len, f8 const eb, u2 const radius, void* stream);
};

template <typename T, class PC>
struct GPU_x_lorenzo_nd {
  static int kernel(
      typename PC::Eq* const in_eq, T* const in_outlier, T* const out_data, psz_len const len,
      f8 const eb, u2 const radius, void* stream);
};

template <typename TIN, typename TOUT, bool ReverseProcess>
int GPU_lorenzo_prequant(
    TIN* const in, size_t const len, f8 const ebx2, f8 const ebx2_r, TOUT* const out,
    void* _stream);

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

template <typename T, typename E, typename FP = T>
int GPU_predict_spline(
    T* in_data, psz_len const data_len3,       //
    E* out_ectrl, psz_len const ectrl_len3,    //
    T* out_anchor, psz_len const anchor_len3,  //
    void* out_outlier,                         //
    f8 const ebx2, f8 const eb_r, uint32_t radius, void* stream);

template <typename T, typename E, typename FP = T>
int GPU_reverse_predict_spline(
    E* in_ectrl, psz_len const ectrl_len3,    //
    T* in_anchor, psz_len const anchor_len3,  //
    T* out_xdata, psz_len const xdata_len3,   //
    f8 const ebx2, f8 const eb_r, uint32_t radius, void* stream);

template <typename T, typename E, typename Fp = T>
struct GPU_spline_construct {
  static int null() { return PSZ_ABORT_NO_SUCH_PREDICTOR; }
  static int kernel_v1(
      T* data, psz_len const data_len3, T* anchor, psz_len const anchor_len3, E* ectrl,
      void* _outlier, double eb, double rel_eb, uint32_t radius, INTERPOLATION_PARAMS& intp_param,
      T* d_profiled_errors, T* h_profiled_errors, u4 const pe_len, void* stream);
};

template <typename T, typename E, typename Fp = T>
struct GPU_spline_reconstruct {
  static int null() { return PSZ_ABORT_NO_SUCH_PREDICTOR; }
  static int kernel_v1(
      T* anchor, psz_len const anchor_len3, E* ectrl, T* xdata, psz_len const xdata_len3,
      T* outlier_tmp, double eb, uint32_t radius, INTERPOLATION_PARAMS intp_param, void* stream);
};

};  // namespace psz::module

#endif /* PSZ_KERNEL_PREDITOR_HH */
