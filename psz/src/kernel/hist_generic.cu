// deps
#include "cusz/type.h"
// definitions
#include "detail/hist.cuhip.inl"

#define INIT_HIST_CUDA(E)                                                                  \
  template void psz::module::GPU_histogram_generic_optimizer_on_initialization<E>(         \
      size_t const data_len, uint16_t const hist_len, int& grid_dim, int& block_dim,       \
      int& shmem_use, int& r_per_block);                                                   \
  template int psz::module::GPU_histogram_generic<E>(                                      \
      E * in_data, size_t const data_len, uint32_t* out_hist, uint16_t const hist_len,     \
      int const grid_dim, int const block_dim, int const shmem_use, int const r_per_block, \
      void* stream);

INIT_HIST_CUDA(u1);
INIT_HIST_CUDA(u2);
INIT_HIST_CUDA(u4);
// INIT_HIST_CUDA(f4);

#undef INIT_HIST_CUDA
