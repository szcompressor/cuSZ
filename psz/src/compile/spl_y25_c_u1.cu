#include "kernel/spl_y25_c.cuh"

template struct psz::module::GPU_c_spline_y25<psz::PredictorTyping<f4, u1>>;
// used too much shared memory
// template struct psz::module::GPU_c_spline_y25<psz::PredictorTyping<f8, u1>>;
