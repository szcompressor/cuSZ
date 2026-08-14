#include "hfr-pbkc.cuh"

// Variant B: 4Ki (M12, IterLog=1), internally 2 iters of M11
// r0 is excluded to avoid BlockDim=2048
template struct phf::module::HFR_PBKC_encode<u1, 12, 1, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u2, 12, 1, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u4, 12, 1, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u1, 12, 2, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u2, 12, 2, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u4, 12, 2, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u1, 12, 3, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u2, 12, 3, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u4, 12, 3, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u1, 12, 4, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u2, 12, 4, u4, 128, 1>;
template struct phf::module::HFR_PBKC_encode<u4, 12, 4, u4, 128, 1>;
