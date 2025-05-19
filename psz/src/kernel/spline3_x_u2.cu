#include "spline3_x.cu"

template struct psz::module::GPU_spline_reconstruct<f4, u2>;
// used too much shared memory
// template struct psz::module::GPU_spline_reconstruct<f8, u2>;