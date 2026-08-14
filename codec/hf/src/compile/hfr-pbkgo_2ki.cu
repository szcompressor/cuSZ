#include "hfr-pbkgo.cuh"

// 2Ki (Magnitude=11): BlockDim = 2^(11-RT); PBKGO runs r2+ (matches its dispatch).
template struct phf::module::HFR_PBKGO_encode<u1, 11, 2, u4, 128>;
template struct phf::module::HFR_PBKGO_encode<u2, 11, 2, u4, 128>;
template struct phf::module::HFR_PBKGO_encode<u4, 11, 2, u4, 128>;
template struct phf::module::HFR_PBKGO_encode<u1, 11, 3, u4, 128>;
template struct phf::module::HFR_PBKGO_encode<u2, 11, 3, u4, 128>;
template struct phf::module::HFR_PBKGO_encode<u4, 11, 3, u4, 128>;
template struct phf::module::HFR_PBKGO_encode<u1, 11, 4, u4, 128>;
template struct phf::module::HFR_PBKGO_encode<u2, 11, 4, u4, 128>;
template struct phf::module::HFR_PBKGO_encode<u4, 11, 4, u4, 128>;
