#include "detail/histsp.cuhip.inl"

#define INIT_HISTSP_CUDA(E)                                                            \
  template int psz::module::GPU_histogram_Cauchy<E>(                                   \
      E * in_data, size_t const data_len, uint32_t* out_hist, uint16_t const hist_len, \
      void* stream);

INIT_HISTSP_CUDA(uint8_t);
INIT_HISTSP_CUDA(uint16_t);
INIT_HISTSP_CUDA(uint32_t);
INIT_HISTSP_CUDA(float);

#undef INIT_HISTSP_CUDA