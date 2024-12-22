#include "detail/histsp.cuhip.inl"

#define INIT_HISTSP_HIP(E)                                    \
  template int psz::module::GPU_histogram_Cauchy<E>(          \
      E * in_data, size_t const data_len, uint32_t* out_hist, \
      uint16_t const hist_len, float* milliseconds, void* stream);

INIT_HISTSP_HIP(uint8_t);
INIT_HISTSP_HIP(uint16_t);
INIT_HISTSP_HIP(uint32_t);
INIT_HISTSP_HIP(float);

#undef INIT_HISTSP_CUDA