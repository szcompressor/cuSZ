// TODO put u? in hf path
#include "hfcxx_module.cu_hip.inl"

namespace phf::cu_hip {

template class modules<u1, u4, true>;
template class modules<u2, u4, true>;
template class modules<u4, u4, true>;

template class modules<u1, u4, false>;
template class modules<u2, u4, false>;
template class modules<u4, u4, false>;

}  // namespace phf::coarse
