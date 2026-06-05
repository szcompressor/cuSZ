// CUDA-side firewall: minimal __global__ + <<<>>> launch for the
// CUDA_MODULE_LOADING=EAGER preload trick. buf_comp.cc stays non-CUDA.

namespace {
__global__ void _Buf_Comp_dummy_kernel() {}
}

namespace psz::buf_comp_dummy {
void launch() { _Buf_Comp_dummy_kernel<<<1, 1>>>(); }
}  // namespace psz::buf_comp_dummy
