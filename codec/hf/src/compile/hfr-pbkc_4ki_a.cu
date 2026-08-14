#include "hfr-pbkc.cuh"

// Variant A: 4Ki (M12, IterLog=2), internally 4 iters of M10
template struct phf::module::HFR_PBKC_encode<u1, 12, 0, u4, 128, 2>;
template struct phf::module::HFR_PBKC_encode<u2, 12, 0, u4, 128, 2>;
template struct phf::module::HFR_PBKC_encode<u4, 12, 0, u4, 128, 2>;
template struct phf::module::HFR_PBKC_encode<u1, 12, 1, u4, 128, 2>;
template struct phf::module::HFR_PBKC_encode<u2, 12, 1, u4, 128, 2>;
template struct phf::module::HFR_PBKC_encode<u4, 12, 1, u4, 128, 2>;
template struct phf::module::HFR_PBKC_encode<u1, 12, 2, u4, 128, 2>;
template struct phf::module::HFR_PBKC_encode<u2, 12, 2, u4, 128, 2>;
template struct phf::module::HFR_PBKC_encode<u4, 12, 2, u4, 128, 2>;
template struct phf::module::HFR_PBKC_encode<u1, 12, 3, u4, 128, 2>;
template struct phf::module::HFR_PBKC_encode<u2, 12, 3, u4, 128, 2>;
template struct phf::module::HFR_PBKC_encode<u4, 12, 3, u4, 128, 2>;
template struct phf::module::HFR_PBKC_encode<u1, 12, 4, u4, 128, 2>;
template struct phf::module::HFR_PBKC_encode<u2, 12, 4, u4, 128, 2>;
template struct phf::module::HFR_PBKC_encode<u4, 12, 4, u4, 128, 2>;
