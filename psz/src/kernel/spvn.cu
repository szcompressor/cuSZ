#include "c_type.h"
#include "detail/spvn.cuhip.inl"
#include "kernel/spvn.hh"

template struct psz::module::GPU_scatter<f4, u4>;
template struct psz::module::GPU_scatter<f8, u4>;