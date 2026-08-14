#include "hfr_encode.cuh"

// 4Ki (Magnitude=12): BlockDim = 2^(12-RT); r0/r1 excluded (>1024 launch cap).
template struct phf::module::HFR_encoder<u1, 12, 2, false, u4>;
template struct phf::module::HFR_encoder<u2, 12, 2, false, u4>;
template struct phf::module::HFR_encoder<u4, 12, 2, false, u4>;
template struct phf::module::HFR_encoder<u1, 12, 3, false, u4>;
template struct phf::module::HFR_encoder<u2, 12, 3, false, u4>;
template struct phf::module::HFR_encoder<u4, 12, 3, false, u4>;
template struct phf::module::HFR_encoder<u1, 12, 4, false, u4>;
template struct phf::module::HFR_encoder<u2, 12, 4, false, u4>;
template struct phf::module::HFR_encoder<u4, 12, 4, false, u4>;
