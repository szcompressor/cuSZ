#include "spline3_c.cu"

template struct psz::module::GPU_spline_construct<f4, u2>;
// used too much shared memory
// template struct psz::module::GPU_spline_construct<f8, u2>;
