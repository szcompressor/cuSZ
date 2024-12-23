// deps
#include "cusz/type.h"
// definitions
#include "detail/hist.cuhip.inl"

#define INIT_HIST_CUDA(E)                                                              \
  template int psz::module::GPU_histogram_generic<E>(                                  \
      E * in_data, size_t const data_len, uint32_t* out_hist, uint16_t const hist_len, \
      float* milliseconds, void* stream);

INIT_HIST_CUDA(u1);
INIT_HIST_CUDA(u2);
INIT_HIST_CUDA(u4);
INIT_HIST_CUDA(f4);

#undef INIT_HIST_CUDA
