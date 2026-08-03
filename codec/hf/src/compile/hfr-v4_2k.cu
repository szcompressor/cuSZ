#include "hfr-pbkc.cuh"

// 2Ki (Magnitude=11): BlockDim = 2^(11-RT); r0 excluded (2048 > launch cap).
template struct phf::module::HFR_V4_encode<u1, 11, 1, u4, 128>;
template struct phf::module::HFR_V4_encode<u2, 11, 1, u4, 128>;
template struct phf::module::HFR_V4_encode<u4, 11, 1, u4, 128>;
template struct phf::module::HFR_V4_encode<u1, 11, 2, u4, 128>;
template struct phf::module::HFR_V4_encode<u2, 11, 2, u4, 128>;
template struct phf::module::HFR_V4_encode<u4, 11, 2, u4, 128>;
template struct phf::module::HFR_V4_encode<u1, 11, 3, u4, 128>;
template struct phf::module::HFR_V4_encode<u2, 11, 3, u4, 128>;
template struct phf::module::HFR_V4_encode<u4, 11, 3, u4, 128>;
template struct phf::module::HFR_V4_encode<u1, 11, 4, u4, 128>;
template struct phf::module::HFR_V4_encode<u2, 11, 4, u4, 128>;
template struct phf::module::HFR_V4_encode<u4, 11, 4, u4, 128>;
