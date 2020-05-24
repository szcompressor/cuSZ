// jtian 20-04-10

#ifndef CANONICAL_CUH
#define CANONICAL_CUH

#include <cstdint>

namespace GPU {

__device__ int max_bw;

template <typename T, typename K>
__global__ void GetCanonicalCode(uint8_t* singleton, int DICT_SIZE);

}  // namespace GPU
#endif
