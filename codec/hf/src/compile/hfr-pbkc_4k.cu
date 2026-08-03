#include "hfr-pbkc.cuh"

// 4Ki variant A: Magnitude=12, IterLog=2 -> BlockDim held at the Magnitude=10 value; Iters = 4.
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

// 4Ki variant B: Magnitude=12, IterLog=1 -> BlockDim doubled (2x the Magnitude=11 grid); Iters = 2.
// r0 is excluded: BlockDim would be 2048 > CUDA's 1024/block limit.
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
