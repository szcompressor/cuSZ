// TODO put u? in hf path
#include "hfcxx_module.cu_hip.inl"

namespace phf::coarse {

template class kernel_wrapper<u1, u4, true>;
template class kernel_wrapper<u2, u4, true>;
template class kernel_wrapper<u4, u4, true>;

template class kernel_wrapper<u1, u4, false>;
template class kernel_wrapper<u2, u4, false>;
template class kernel_wrapper<u4, u4, false>;

}  // namespace phf::coarse
