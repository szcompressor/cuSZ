#include "scatter_cxx.cuhip.inl"

INSTANTIATE_PSZCXX_MODULE_SCATTER(CUDA, f4, true)
INSTANTIATE_PSZCXX_MODULE_SCATTER(CUDA, f4, false)

#undef INSTANTIATE_PSZCXX_MODULE_SCATTER