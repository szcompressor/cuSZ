// TODO put u? in hf path
#include "hfcxx_module.cuhip.inl"

namespace phf::cuhip {

template class modules<u1, u4, true>;
template class modules<u2, u4, true>;
template class modules<u4, u4, true>;

template class modules<u1, u4, false>;
template class modules<u2, u4, false>;
template class modules<u4, u4, false>;

}  // namespace phf::coarse
