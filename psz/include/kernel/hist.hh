#ifndef PSZ_MODULE_HIST_HH
#define PSZ_MODULE_HIST_HH

#include "cusz/type.h"

namespace psz::module {

template <typename E>
int SEQ_histogram_generic(
    E* in_data, size_t const data_len, uint32_t* out_hist,
    uint16_t const hist_len, float* milliseconds);

template <typename E>
int GPU_histogram_generic(
    E* in_data, size_t const data_len, uint32_t* out_hist,
    uint16_t const hist_len, float* milliseconds, void* stream);

template <typename E>
int SEQ_histogram_Cauchy_v2(
    E* in_data, size_t const data_len, uint32_t* out_hist,
    uint16_t const hist_len, float* milliseconds);

template <typename E>
int GPU_histogram_Cauchy(
    E* in_data, size_t const data_len, uint32_t* out_hist,
    uint16_t const hist_len, float* milliseconds, void* stream);

}  // namespace psz::module

#endif /* PSZ_MODULE_HIST_HH */
