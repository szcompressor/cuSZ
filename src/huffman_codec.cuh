#ifndef NEW_ENCODE_CUH
#define NEW_ENCODE_CUH

namespace prototype {
template <typename T, typename Q>
__global__ void EncodeFixedLen(T* data, Q* hcoded, size_t data_len, Q* codebook) {
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= data_len) return;
    hcoded[gid] = codebook[data[gid]];  // try to exploit cache?
    __syncthreads();
}
}  // namespace prototype

#endif
