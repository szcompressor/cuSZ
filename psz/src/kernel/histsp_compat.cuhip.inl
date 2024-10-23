/**
 * @file histsp.cuhip.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-05-18
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <cstdint>

#include "detail/histsp.cuhip.inl"
#include "module/cxx_module.hh"


////////////////////////////////////////////////////////////////////////////////
#define SPECIALIZE_PSZCXX_COMPAT_MODULE_HIST_CAUCHY(BACKEND, E)               \
  template <>                                                                 \
  int pszcxx_compat_histogram_cauchy<BACKEND, E, uint32_t>(                   \
      E * in_data, uint32_t data_len, uint32_t * out_hist, uint32_t hist_len, \
      float* milliseconds, void* stream)                                      \
  {                                                                           \
    return psz::cuhip::GPU_histogram_sparse<E, uint32_t>(                     \
        in_data, data_len, out_hist, hist_len, milliseconds,                  \
        (cudaStream_t)stream);                                                \
  }
