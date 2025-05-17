#include "detail/lrz_x.cuhip.inl"

using psz::Toggle;

template struct psz::module::GPU_x_lorenzo_nd<
    float, psz::PredConfig<float, psz::PredFunc<Toggle::ZigZagDisabled>>>;

template struct psz::module::GPU_x_lorenzo_nd<
    double, psz::PredConfig<double, psz::PredFunc<Toggle::ZigZagDisabled>>>;

template struct psz::module::GPU_x_lorenzo_nd<
    float, psz::PredConfig<float, psz::PredFunc<Toggle::ZigZagEnabled>>>;

template struct psz::module::GPU_x_lorenzo_nd<
    double, psz::PredConfig<double, psz::PredFunc<Toggle::ZigZagEnabled>>>;