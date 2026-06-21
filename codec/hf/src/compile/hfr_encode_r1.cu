#include "hfr_encode.cuh"

template int phf::module::HFR_encoder<u1, 10, 1, false, u4>::template GPU_kernel_v2<128>(
    u1*, size_t, u4*, u4*, psz::_future::bheader<u1, 128>*, void*, RMerge, SMerge);
template int phf::module::HFR_encoder<u2, 10, 1, false, u4>::template GPU_kernel_v2<128>(
    u2*, size_t, u4*, u4*, psz::_future::bheader<u2, 128>*, void*, RMerge, SMerge);
