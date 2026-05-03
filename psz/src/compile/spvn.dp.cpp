#include "kernel/spvn.dp.cpp"

#include "c_type.h"
#include "kernel.hh"

template struct psz::module::GPU_scatter<f4, u4>;
template struct psz::module::GPU_scatter<f8, u4>;
