#include "kernel/spl_y25_c.cuh"

template struct psz::module::GPU_c_spline_y25<psz::PredictorTyping<f4, u4>, psz::PredictorFeature<0b0>>;
// TODO used too much shared memory
// template struct psz::module::GPU_c_spline_y25<psz::PredictorTyping<f8, u4>, psz::PredictorFeature<0b0>>;
