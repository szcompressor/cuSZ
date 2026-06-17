// TODO put u? in hf path
#include "hf_kernels.cu.inl"

namespace phf::cuhip {

template class modules<u1, u4>;
template class modules<u2, u4>;
template class modules<u4, u4>;

template void modules<u2, u4>::GPU_coarse_decode<u2>(
    u4*, uint8_t*, size_t const, u4*, u4*, size_t const, size_t const, u2*, uint8_t*, void*);

template void modules<u2, u4>::GPU_coarse_decode<f4>(
    u4*, uint8_t*, size_t const, u4*, u4*, size_t const, size_t const, f4*, uint8_t*, void*);
template void modules<u2, u4>::GPU_coarse_decode<f8>(
    u4*, uint8_t*, size_t const, u4*, u4*, size_t const, size_t const, f8*, uint8_t*, void*);

}  // namespace phf::cuhip
