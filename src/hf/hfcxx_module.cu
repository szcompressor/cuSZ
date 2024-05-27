// TODO put u? in hf path
#include "hfcxx_module.cu_hip.inl"

template class _2403::phf_kernel_wrapper<u1, u4, u4, true>;
template class _2403::phf_kernel_wrapper<u2, u4, u4, true>;
template class _2403::phf_kernel_wrapper<u4, u4, u4, true>;

template class _2403::phf_kernel_wrapper<u1, u4, u4, false>;
template class _2403::phf_kernel_wrapper<u2, u4, u4, false>;
template class _2403::phf_kernel_wrapper<u4, u4, u4, false>;
