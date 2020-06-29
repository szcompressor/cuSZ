//
// By Cody Rivera, 6/2020
//

#ifndef PAR_HUFFMAN_CUH
#define PAR_HUFFMAN_CUH

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cooperative_groups.h>

using namespace std;
using namespace cooperative_groups;

// Helper kernels
template <typename T>
__global__ void GPU_FillArraySequence(T* array, unsigned int size);
template <typename T>
__global__ void GPU_GetFirstNonzeroIndex(T* array, unsigned int size, unsigned int* result);
template <typename T, typename Q>
__global__ void GPU_ReorderByIndex(T* array, Q* index, unsigned int size);

// Parallel huffman global memory and kernels
namespace parHuff {
// GenerateCL Locals
__device__ int iNodesFront = 0;
__device__ int iNodesRear = 0;
__device__ int lNodesCur = 0;

__device__ int iNodesSize = 0;
__device__ int curLeavesNum;

__device__ int minFreq;

__device__ int tempLength;

__device__ int mergeFront;
__device__ int mergeRear;

__device__ int lNodesIndex;

// GenerateCW Locals
__device__ int CCL;
__device__ int CDPI;

// Profiling
__device__ long long int s[10];
__device__ long long int st[10];

// Codeword length
template <typename F>
__global__ void GPU_GenerateCL(F* histogram, F* CL, int size,
    /* Global Arrays */
    F* lNodesFreq, int* lNodesLeader,
    F* iNodesFreq, int* iNodesLeader,
    F* tempFreq, int* tempIsLeaf, int* tempIndex,
    F* copyFreq, int* copyIsLeaf, int* copyIndex,
    uint32_t* diagonal_path_intersections, int mblocks, int mthreads);

// Forward Codebook
template<typename F, typename H>
__global__ void GPU_GenerateCW(F* CL, H* CW, int size);
}

template <typename H>
void ParGetCodebook(int stateNum, unsigned int* freq, H* codebook);

#endif
