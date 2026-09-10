#include "c_type.h"
#include "kernel.hh"
#include "kernel/widen.cu"

template struct psz::module::GPU_widen<f4, u2>;
template struct psz::module::GPU_widen<f8, u2>;
