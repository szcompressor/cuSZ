#include "kernel/lrz_x.cuh"

namespace psz {

using TypesF4 = PredictorTyping<float>;
using TypesF8 = PredictorTyping<double>;
using TypesF4Eq4 = PredictorTyping<float, u4>;
using TypesF8Eq4 = PredictorTyping<double, u4>;

using FeaturesCompat = PredictorFeature<0b0>;
using FeaturesZigZag = PredictorFeature<0b1>;

template struct psz::module::GPU_x_lorenzo_nd<TypesF4, FeaturesCompat>;
template struct psz::module::GPU_x_lorenzo_nd<TypesF8, FeaturesCompat>;
template struct psz::module::GPU_x_lorenzo_nd<TypesF4, FeaturesZigZag>;
template struct psz::module::GPU_x_lorenzo_nd<TypesF8, FeaturesZigZag>;

template struct psz::module::GPU_x_lorenzo_nd<TypesF4Eq4, FeaturesCompat>;
template struct psz::module::GPU_x_lorenzo_nd<TypesF8Eq4, FeaturesCompat>;
template struct psz::module::GPU_x_lorenzo_nd<TypesF4Eq4, FeaturesZigZag>;
template struct psz::module::GPU_x_lorenzo_nd<TypesF8Eq4, FeaturesZigZag>;

}  // namespace psz