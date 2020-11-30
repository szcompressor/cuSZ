/**
 *  @file Huffman.cu
 *  @author Sheng Di
 *  Modified by Jiannan Tian
 *  @date Jan. 7, 2020
 *  Created on Aug., 2016
 *  @brief Customized Huffman Encoding, Compression and Decompression functions.
 *         Also modified for GPU prototyping.
 *  (C) 2016 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "huffman.cuh"

using namespace std;
template <typename T, typename Q>
__global__ void prototype::GPU_Histogram(T* input_data, Q* output, size_t N, int symbols_per_thread)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int j;
    if (i * symbols_per_thread < N) {  // if there is a symbol to count
        for (j = i * symbols_per_thread; j < (i + 1) * symbols_per_thread; j++) {
            if (j < N) {
                unsigned int item = input_data[j];  // Symbol to count
                atomicAdd(&output[item], 1);        // update bin count by 1
            }
        }
    }
}

template <typename T, typename Q>
__global__ void prototype::EncodeFixedLen(T* data, Q* hcoded, size_t data_len, Q* codebook)
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= data_len) return;
    hcoded[gid] = codebook[data[gid]];  // try to exploit cache?
    __syncthreads();
}

// auxiliary functions done
__host__ HuffmanTree* createHuffmanTreeCPU(int stateNum)
{
    auto ht = (HuffmanTree*)malloc(sizeof(HuffmanTree));
    memset(ht, 0, sizeof(HuffmanTree));
    ht->stateNum = stateNum;
    ht->allNodes = 2 * stateNum;

    ht->pool = (struct node_t*)malloc(ht->allNodes * 2 * sizeof(struct node_t));
    ht->qqq  = (node_list*)malloc(ht->allNodes * 2 * sizeof(node_list));
    ht->code = (uint64_t**)malloc(ht->stateNum * sizeof(uint64_t*));
    ht->cout = (uint8_t*)malloc(ht->stateNum * sizeof(uint8_t));

    memset(ht->pool, 0, ht->allNodes * 2 * sizeof(struct node_t));
    memset(ht->qqq, 0, ht->allNodes * 2 * sizeof(node_list));
    memset(ht->code, 0, ht->stateNum * sizeof(uint64_t*));
    memset(ht->cout, 0, ht->stateNum * sizeof(uint8_t));
    ht->qq      = ht->qqq - 1;
    ht->n_nodes = 0;
    ht->n_inode = 0;
    ht->qend    = 1;

    return ht;
}

__device__ HuffmanTree* createHuffmanTreeGPU(int stateNum)
{
    auto ht = (HuffmanTree*)malloc(sizeof(HuffmanTree));
    memset(ht, 0, sizeof(HuffmanTree));
    ht->stateNum = stateNum;
    ht->allNodes = 2 * stateNum;

    ht->pool = (struct node_t*)malloc(ht->allNodes * 2 * sizeof(struct node_t));
    ht->qqq  = (node_list*)malloc(ht->allNodes * 2 * sizeof(node_list));
    ht->code = (uint64_t**)malloc(ht->stateNum * sizeof(uint64_t*));
    ht->cout = (uint8_t*)malloc(ht->stateNum * sizeof(uint8_t));

    memset(ht->pool, 0, ht->allNodes * 2 * sizeof(struct node_t));
    memset(ht->qqq, 0, ht->allNodes * 2 * sizeof(node_list));
    memset(ht->code, 0, ht->stateNum * sizeof(uint64_t*));
    memset(ht->cout, 0, ht->stateNum * sizeof(uint8_t));
    ht->qq      = ht->qqq - 1;
    ht->n_nodes = 0;
    ht->n_inode = 0;
    ht->qend    = 1;

    return ht;
}

__host__ __device__ node_list new_node(HuffmanTree* huffmanTree, size_t freq, uint32_t c, node_list a, node_list b)
{
    node_list n = huffmanTree->pool + huffmanTree->n_nodes++;
    if (freq) {
        n->c    = c;
        n->freq = freq;
        n->t    = 1;
    }
    else {
        n->left  = a;
        n->right = b;
        n->freq  = a->freq + b->freq;
        n->t     = 0;
        // n->c = 0;
    }
    return n;
}

/* priority queue */
__host__ __device__ void qinsert(HuffmanTree* ht, node_list n)
{
    int j, i = ht->qend++;
    while ((j = (i >> 1))) {  // j=i/2
        if (ht->qq[j]->freq <= n->freq) break;
        ht->qq[i] = ht->qq[j], i = j;
    }
    ht->qq[i] = n;
}

__host__ __device__ node_list qremove(HuffmanTree* ht)
{
    int       i, l;
    node_list n = ht->qq[i = 1];

    if (ht->qend < 2) return 0;
    ht->qend--;
    while ((l = (i << 1)) < ht->qend) {  // l=(i*2)
        if (l + 1 < ht->qend && ht->qq[l + 1]->freq < ht->qq[l]->freq) l++;
        ht->qq[i] = ht->qq[l], i = l;
    }
    ht->qq[i] = ht->qq[ht->qend];
    return n;
}

/* walk the tree and put 0s and 1s */
/**
 * @out1 should be set to 0.
 * @out2 should be 0 as well.
 * @index: the index of the byte
 * */
__host__ __device__ void build_code(HuffmanTree* ht, node_list n, int len, uint64_t out1, uint64_t out2)
{
    if (n->t) {
        ht->code[n->c] = (uint64_t*)malloc(2 * sizeof(uint64_t));
        if (len <= 64) {
            (ht->code[n->c])[0] = out1 << (64 - len);
            (ht->code[n->c])[1] = out2;
        }
        else {
            (ht->code[n->c])[0] = out1;
            (ht->code[n->c])[1] = out2 << (128 - len);
        }
        ht->cout[n->c] = (uint8_t)len;
        return;
    }

    int index = len >> 6;  //=len/64
    if (index == 0) {
        out1 = out1 << 1;
        out1 = out1 | 0;
        build_code(ht, n->left, len + 1, out1, 0);
        out1 = out1 | 1;
        build_code(ht, n->right, len + 1, out1, 0);
    }
    else {
        if (len % 64 != 0) out2 = out2 << 1;
        out2 = out2 | 0;
        build_code(ht, n->left, len + 1, out1, out2);
        out2 = out2 | 1;
        build_code(ht, n->right, len + 1, out1, out2);
    }
}

////////////////////////////////////////////////////////////////////////////////
// internal functions
////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ node_list top(internal_stack_t* s)
{
    return s->_a[s->depth - 1];
}

template <typename T>
__device__ __forceinline__ void push_v2(internal_stack_t* s, node_list n, T path, T len)
{
    if (s->depth + 1 <= MAX_DEPTH) {
        s->depth += 1;

        s->_a[s->depth - 1]           = n;
        s->saved_path[s->depth - 1]   = path;
        s->saved_length[s->depth - 1] = len;
    }
    else
        printf("Error: stack overflow\n");
}

__device__ __forceinline__ bool isEmpty(internal_stack_t* s)
{
    return (s->depth == 0);
}

// TODO check with typing
template <typename T>
__device__ __forceinline__ node_list pop_v2(internal_stack_t* s, T* path_to_restore, T* length_to_restore)
{
    node_list n;

    if (isEmpty(s)) {
        printf("Error: stack underflow, exiting...\n");
        return nullptr;
        //        exit(0);
    }
    else {
        // TODO holding array -> __a
        n                   = s->_a[s->depth - 1];
        s->_a[s->depth - 1] = nullptr;

        *length_to_restore = s->saved_length[s->depth - 1];
        *path_to_restore   = s->saved_path[s->depth - 1];
        s->depth -= 1;

        return n;
    }
}

template <typename Q>
__device__ void InOrderTraverse_v2(HuffmanTree* ht, Q* codebook)
{
    node_list root = ht->qq[1];
    auto      s    = new internal_stack_t();

    bool done = 0;
    Q    out1 = 0, len = 0;

    while (!done) {
        if (root->left or root->right) {
            push_v2(s, root, out1, len);
            root = root->left;
            out1 <<= 1u;
            out1 |= 0u;
            len += 1;
        }
        else {
            uint32_t bincode  = root->c;
            codebook[bincode] = out1 | ((len & (Q)0xffu) << (sizeof(Q) * 8 - 8));
            if (!isEmpty(s)) {
                root = pop_v2(s, &out1, &len);
                root = root->right;
                out1 <<= 1u;
                out1 |= 1u;
                len += 1;
            }
            else
                done = true;
        }
    } /* end of while */
}

template <typename H>
__global__ void InitHuffTreeAndGetCodebook(int stateNum, unsigned int* freq, H* codebook)
{  // length known as huffmanTree->allNodes
    if (threadIdx.x != 0) return;
    global_gpuTree = createHuffmanTreeGPU(stateNum);
    for (size_t i = 0; i < global_gpuTree->allNodes; i++)
        if (freq[i]) qinsert(global_gpuTree, new_node(global_gpuTree, freq[i], i, 0, 0));
    while (global_gpuTree->qend > 2)
        qinsert(global_gpuTree, new_node(global_gpuTree, 0, 0, qremove(global_gpuTree), qremove(global_gpuTree)));
    InOrderTraverse_v2<H>(global_gpuTree, codebook);
}

// TODO `unsigned int` seems trivial to pick up
template __global__ void InitHuffTreeAndGetCodebook<uint32_t>(int stateNum, unsigned int* freq, uint32_t* codebook);
template __global__ void InitHuffTreeAndGetCodebook<uint64_t>(int stateNum, unsigned int* freq, uint64_t* codebook);
