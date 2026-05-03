#include "c_type.h"
#include "kernel/spvn.cu"
#include "kernel.hh"

template struct psz::module::GPU_scatter<f4, u4>;
template struct psz::module::GPU_scatter<f8, u4>;
template struct psz::module::GPU_scatter<u2, u4>;