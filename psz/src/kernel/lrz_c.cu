#include "detail/lrz_c.cuhip.inl"
#include "mem/buf_comp.hh"

using psz::Toggle;

template struct psz::module::GPU_c_lorenzo_nd<
    float, psz::PredConfig<float, psz::PredFunc<Toggle::ZigZagDisabled>>, psz::Buf_Comp<float>>;

template struct psz::module::GPU_c_lorenzo_nd<
    double, psz::PredConfig<double, psz::PredFunc<Toggle::ZigZagDisabled>>, psz::Buf_Comp<double>>;

template struct psz::module::GPU_c_lorenzo_nd<
    float, psz::PredConfig<float, psz::PredFunc<Toggle::ZigZagEnabled>>, psz::Buf_Comp<float>>;

template struct psz::module::GPU_c_lorenzo_nd<
    double, psz::PredConfig<double, psz::PredFunc<Toggle::ZigZagEnabled>>, psz::Buf_Comp<double>>;
