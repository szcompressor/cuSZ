#include "kernel/lrz_x.cuh"

namespace psz {

using TypesF4 = PredictorTyping<float>;
using TypesF8 = PredictorTyping<double>;

using FeaturesDefault = PredictorFeature<Toggle::ZigZag_Off, Toggle::H1L_Off, Toggle::H1G_Off>;
using FeaturesZigZag = PredictorFeature<Toggle::ZigZag_On, Toggle::H1L_Off, Toggle::H1G_Off>;

template struct psz::module::GPU_x_lorenzo_nd<TypesF4, FeaturesDefault>;
template struct psz::module::GPU_x_lorenzo_nd<TypesF8, FeaturesDefault>;
template struct psz::module::GPU_x_lorenzo_nd<TypesF4, FeaturesZigZag>;
template struct psz::module::GPU_x_lorenzo_nd<TypesF8, FeaturesZigZag>;

}  // namespace psz