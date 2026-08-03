#include "hfr-pbkgo.cuh"

// 4Ki (Magnitude=12): BlockDim = 2^(12-RT); PBKGO runs r2+ (r2 -> 1024 threads).
template struct phf::module::HFR_PBKGO_encode<u1, 12, 2, u4, 128>;
template struct phf::module::HFR_PBKGO_encode<u2, 12, 2, u4, 128>;
template struct phf::module::HFR_PBKGO_encode<u4, 12, 2, u4, 128>;
template struct phf::module::HFR_PBKGO_encode<u1, 12, 3, u4, 128>;
template struct phf::module::HFR_PBKGO_encode<u2, 12, 3, u4, 128>;
template struct phf::module::HFR_PBKGO_encode<u4, 12, 3, u4, 128>;
template struct phf::module::HFR_PBKGO_encode<u1, 12, 4, u4, 128>;
template struct phf::module::HFR_PBKGO_encode<u2, 12, 4, u4, 128>;
template struct phf::module::HFR_PBKGO_encode<u4, 12, 4, u4, 128>;
