namespace {
__global__ void _Buf_HF_dummy_kernel() {}
}

namespace phf::_dummy {
void launch() { _Buf_HF_dummy_kernel<<<1, 1>>>(); }
}  // namespace phf::_dummy
