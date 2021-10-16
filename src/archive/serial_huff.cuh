/**
 *  @file Huffman.cuh
 *  @author Sheng Di
 *  Modified by Jiannan Tian
 *  @date Jan. 7, 2020
 *  Created on Aug., 2016
 *  @brief Customized Huffman Encoding, Compression and Decompression functions.
 *         Also modified for GPU prototyping (header).
 *  (C) 2016 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef HUFFMAN_CUH
#define HUFFMAN_CUH

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace std;
namespace prototype {
template <typename Input, typename Output>
__global__ void GPU_Histogram(Input*, Output*, size_t, int);

template <typename Input, typename Huff>
__global__ void EncodeFixedLen(Input*, Huff*, size_t, Huff*);

}  // namespace prototype

struct alignas(8) node_t {
    struct node_t *left, *right;
    size_t         freq;
    char           t;  // in_node:0; otherwise:1
    uint32_t       c;
};

typedef struct node_t* node_list;

typedef struct alignas(8) HuffmanTree {
    uint32_t       stateNum;
    uint32_t       allNodes;
    struct node_t* pool;
    node_list *    qqq, *qq;  // the root node of the HuffmanTree is qq[1]
    int            n_nodes;   // n_nodes is for compression
    int            qend;
    uint64_t**     code;
    uint8_t*       cout;
    int            n_inode;  // n_inode is for decompression
} HuffmanTree;

HuffmanTree* createHuffmanTree(int stateNum);

__host__ __device__ node_list new_node(HuffmanTree* huffmanTree, size_t freq, uint32_t c, node_list a, node_list b);
__host__ __device__ void      qinsert(HuffmanTree* ht, node_list n);
__host__ __device__ node_list qremove(HuffmanTree* ht);
__host__ __device__ void      build_code(HuffmanTree* ht, node_list n, int len, uint64_t out1, uint64_t out2);

// auxiliary functions done
__host__ HuffmanTree* createHuffmanTreeCPU(int stateNum);

__device__ HuffmanTree* createHuffmanTreeGPU(int stateNum);

__host__ __device__ node_list new_node(HuffmanTree* huffmanTree, size_t freq, uint32_t c, node_list a, node_list b);

/* priority queue */
__host__ __device__ void qinsert(HuffmanTree* ht, node_list n);

__host__ __device__ node_list qremove(HuffmanTree* ht);

__host__ __device__ void build_code(HuffmanTree* ht, node_list n, int len, uint64_t out1, uint64_t out2);

////////////////////////////////////////////////////////////////////////////////
// internal functions
////////////////////////////////////////////////////////////////////////////////

const int MAX_DEPTH = 32;
//#define MAX_DEPTH 32

typedef struct alignas(8) Stack {
    node_list _a[MAX_DEPTH];
    uint64_t  saved_path[MAX_DEPTH];
    uint64_t  saved_length[MAX_DEPTH];
    uint64_t  depth = 0;
} internal_stack_t;

__device__ __forceinline__ bool isEmpty(internal_stack_t* s);

__device__ __forceinline__ node_list top(internal_stack_t* s);

template <typename T>
__device__ __forceinline__ void push_v2(internal_stack_t* s, node_list n, T path, T len);

// TODO check with typing
template <typename T>
__device__ __forceinline__ node_list pop_v2(internal_stack_t* s, T* path_to_restore, T* length_to_restore);

template <typename Q>
__device__ void InOrderTraverse_v2(HuffmanTree* ht, Q* codebook);

////////////////////////////////////////////////////////////////////////////////
// global functions
////////////////////////////////////////////////////////////////////////////////

__device__ HuffmanTree* global_gpuTree;

template <typename H>
__global__ void InitHuffTreeAndGetCodebook(int stateNum, unsigned int* freq, H* codebook);

#endif
