#include "kernel/spl_y25_x.cuh"

template struct psz::module::GPU_x_spline_y25<psz::PredictorTyping<f4, u1>>;
// used too much shared memory
// template struct psz::module::GPU_x_spline_y25<psz::PredictorTyping<f8, u1>>;