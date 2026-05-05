#include "kernel/lrz_c.cu.inl"

namespace psz {

using TypesF4 = PredictorTyping<float>;
using TypesF8 = PredictorTyping<double>;

using FeaturesDefault = PredictorFeature<Toggle::ZigZag_Off, Toggle::H1L_Off, Toggle::H1G_Off>;
using FeaturesZigZag = PredictorFeature<Toggle::ZigZag_On, Toggle::H1L_Off, Toggle::H1G_Off>;
using FeaturesLean = PredictorFeature<Toggle::ZigZag_Off, Toggle::H1L_On, Toggle::H1G_Off>;

template struct module::GPU_c_lorenzo_nd<TypesF4, FeaturesDefault>;
template struct module::GPU_c_lorenzo_nd<TypesF4, FeaturesZigZag>;
template struct module::GPU_c_lorenzo_nd<TypesF4, FeaturesLean>;

template struct module::GPU_c_lorenzo_nd<TypesF8, FeaturesDefault>;
template struct module::GPU_c_lorenzo_nd<TypesF8, FeaturesZigZag>;
template struct module::GPU_c_lorenzo_nd<TypesF8, FeaturesLean>;

};  // namespace psz