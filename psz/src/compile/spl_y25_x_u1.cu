#include "kernel/spl_y25_x.cu"

template struct psz::module::GPU_x_spline_y25<f4, u1>;
// used too much shared memory
// template struct psz::module::GPU_x_spline_y25<f8, u1>;