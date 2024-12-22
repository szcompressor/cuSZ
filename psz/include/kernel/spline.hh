#include <cstddef>
#include <cstdint>

#include "mem/memobj.hh"
using namespace portable;

template <typename T, typename E, typename FP = T>
[[deprecated]] int pszcxx_predict_spline(
    memobj<T>* data, memobj<T>* anchor, memobj<E>* errctrl, void* _outlier,
    double eb, uint32_t radius, float* time, void* stream);

template <typename T, typename E, typename FP = T>
[[deprecated]] int pszcxx_reverse_predict_spline(
    memobj<T>* anchor, memobj<E>* errctrl, memobj<T>* xdata, double eb,
    uint32_t radius, float* time, void* stream);

namespace psz::cuhip {

template <typename T, typename E, typename FP = T>
int GPU_predict_spline(
    T* in_data, dim3 const data_len3, dim3 const data_stride3,         //
    E* out_ectrl, dim3 const ectrl_len3, dim3 const ectrl_stride3,     //
    T* out_anchor, dim3 const anchor_len3, dim3 const anchor_stride3,  //
    void* out_outlier,                                                 //
    double eb, uint32_t radius, float* time, void* stream);

template <typename T, typename E, typename FP = T>
int GPU_reverse_predict_spline(
    E* in_ectrl, dim3 const ectrl_len3, dim3 const ectrl_stride3,     //
    T* in_anchor, dim3 const anchor_len3, dim3 const anchor_stride3,  //
    T* out_xdata, dim3 const xdata_len3, dim3 const xdata_stride3,    //
    double eb, uint32_t radius, float* time, void* stream);

};  // namespace psz::cuhip
