#include "hfr-pbkc.cuh"

// 4Ki (Magnitude=12): BlockDim = 2^(12-RT); r0/r1 excluded (>1024 launch cap).
template struct phf::module::HFR_V4_encode<u1, 12, 2, u4, 128>;
template struct phf::module::HFR_V4_encode<u2, 12, 2, u4, 128>;
template struct phf::module::HFR_V4_encode<u4, 12, 2, u4, 128>;
template struct phf::module::HFR_V4_encode<u1, 12, 3, u4, 128>;
template struct phf::module::HFR_V4_encode<u2, 12, 3, u4, 128>;
template struct phf::module::HFR_V4_encode<u4, 12, 3, u4, 128>;
template struct phf::module::HFR_V4_encode<u1, 12, 4, u4, 128>;
template struct phf::module::HFR_V4_encode<u2, 12, 4, u4, 128>;
template struct phf::module::HFR_V4_encode<u4, 12, 4, u4, 128>;
