#include "hfr_encode.cuh"

// 2Ki (Magnitude=11): BlockDim = 2^(11-RT); r0 excluded (2048 > launch cap).
template struct phf::module::HFR_encoder<u1, 11, 1, false, u4>;
template struct phf::module::HFR_encoder<u2, 11, 1, false, u4>;
template struct phf::module::HFR_encoder<u4, 11, 1, false, u4>;
template struct phf::module::HFR_encoder<u1, 11, 2, false, u4>;
template struct phf::module::HFR_encoder<u2, 11, 2, false, u4>;
template struct phf::module::HFR_encoder<u4, 11, 2, false, u4>;
template struct phf::module::HFR_encoder<u1, 11, 3, false, u4>;
template struct phf::module::HFR_encoder<u2, 11, 3, false, u4>;
template struct phf::module::HFR_encoder<u4, 11, 3, false, u4>;
template struct phf::module::HFR_encoder<u1, 11, 4, false, u4>;
template struct phf::module::HFR_encoder<u2, 11, 4, false, u4>;
template struct phf::module::HFR_encoder<u4, 11, 4, false, u4>;
