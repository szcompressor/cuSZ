#include "c_type.h"
#include "kernel/cast.cu"
#include "kernel.hh"

template struct psz::module::GPU_cast<u2, f4>;
template struct psz::module::GPU_cast<u2, f8>;
template struct psz::module::GPU_cast<u4, f4>;
template struct psz::module::GPU_cast<u4, f8>;
