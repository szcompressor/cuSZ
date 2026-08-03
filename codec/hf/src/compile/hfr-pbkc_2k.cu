#include "hfr-pbkc.cuh"

// 2Ki variant: Magnitude=11, IterLog=1 -> BlockDim held at the Magnitude=10 value; Iters = 2.
template struct phf::module::HFR_PBKC_encode<u1, 11, 0, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u2, 11, 0, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u4, 11, 0, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u1, 11, 1, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u2, 11, 1, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u4, 11, 1, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u1, 11, 2, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u2, 11, 2, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u4, 11, 2, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u1, 11, 3, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u2, 11, 3, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u4, 11, 3, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u1, 11, 4, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u2, 11, 4, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u4, 11, 4, u4, 128, 1>;
