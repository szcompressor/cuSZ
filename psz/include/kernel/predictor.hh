#ifndef PSZ_KERNEL_PREDITOR_HH
#define PSZ_KERNEL_PREDITOR_HH

#include <array>
#include <cstdint>

#include "cusz/type.h"

typedef struct psz_dim3_seq {
  uint32_t x, y, z;
} psz_dim3_seq;

psz_dim3_seq psz_div3(psz_dim3_seq len, psz_dim3_seq sublen);

using stdlen3 = std::array<size_t, 3>;

namespace psz::module {

template <typename T, class PC>
struct GPU_c_lorenzo_nd {
  static int kernel(
      T* const in_data, stdlen3 const _data_len3, typename PC::Eq* const out_eq, void* out_outlier,
      u4* out_top1, f8 const eb, uint16_t const radius, void* stream);
};

template <typename T, class PC>
struct GPU_x_lorenzo_nd {
  static int kernel(
      typename PC::Eq* const in_eq, T* const in_outlier, T* const out_data,
      stdlen3 const _data_len3, f8 const eb, uint16_t const radius, void* stream);
};

template <typename TIN, typename TOUT, bool ReverseProcess>
int GPU_lorenzo_prequant(
    TIN* const in, size_t const len, f8 const ebx2, f8 const ebx2_r, TOUT* const out,
    void* _stream);

}  // namespace psz::module

namespace psz::module {

template <typename T, typename Eq>
struct GPU_PROTO_c_lorenzo_nd_with_outlier {
  static int kernel(
      T* const in_data, std::array<size_t, 3> const data_len3, Eq* const out_eq, void* out_outlier,
      f8 const ebx2, f8 const ebx2_r, uint16_t const radius, void* stream);
};

template <typename T, typename Eq>
struct GPU_PROTO_x_lorenzo_nd {
  static int kernel(
      Eq* in_eq, T* in_outlier, T* out_data, std::array<size_t, 3> const data_len3, f8 const ebx2,
      f8 const ebx2_r, int const radius, void* stream);
};

}  // namespace psz::module

template <typename T, typename Eq>
pszerror CPU_c_lorenzo_nd_with_outlier(
    T* const in_data, psz_dim3_seq const data_len3, Eq* const out_eq, void* out_outlier,
    f8 const eb, uint16_t const radius, float* time_elapsed);

template <typename T, typename Eq>
pszerror CPU_x_lorenzo_nd(
    Eq* const in_eq, T* const in_outlier, T* const out_data, psz_dim3_seq const data_len3,
    f8 const eb, uint16_t const radius, f4* time_elapsed);

namespace psz::module {

template <typename T, typename E, typename Fp = T>
struct GPU_spline_construct {
  //   static int kernel_v0(
  //       memobj<T>* data, memobj<T>* anchor, memobj<E>* errctrl, void* _outlier, double eb,
  //       double rel_eb, uint32_t radius, INTERPOLATION_PARAMS& intp_param, float* time, void*
  //       stream, memobj<T>* profiling_errors);

  static int kernel_v1(
      T* data, stdlen3 const data_len3, T* anchor, stdlen3 const anchor_len3, E* ectrl,
      void* _outlier, double eb, double rel_eb, uint32_t radius, INTERPOLATION_PARAMS& intp_param,
      float* time, T* d_profiling_errors, T* h_profiling_errors, u4 const pe_len, void* stream);
};

template <typename T, typename E, typename Fp = T>
struct GPU_spline_reconstruct {
  //   static int kernel_v0(
  //       memobj<T>* anchor, memobj<E>* errctrl, memobj<T>* xdata, T* outlier_tmp, double eb,
  //       uint32_t radius, INTERPOLATION_PARAMS intp_param, float* time, void* stream);

  static int kernel_v1(
      T* anchor, stdlen3 const anchor_len3, E* ectrl, T* xdata, stdlen3 const xdata_len3,
      T* outlier_tmp, double eb, uint32_t radius, INTERPOLATION_PARAMS intp_param, float* time,
      void* stream);
};

}  // namespace psz::module

#endif /* PSZ_KERNEL_PREDITOR_HH */
