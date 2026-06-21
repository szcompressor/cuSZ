#include "hfr_concat.cuh"

template struct phf::concat_via_scatter_ppc<32>;
template struct phf::concat_via_scatter_ppc<64>;
template struct phf::concat_via_scatter_ppc<128>;
template struct phf::concat_via_scatter_ppc<256>;

template struct phf::_future_concat_via_scatter<u1, 128>;
template struct phf::_future_concat_via_scatter<u2, 128>;
template struct phf::_future_concat_via_scatter<u4, 128>;