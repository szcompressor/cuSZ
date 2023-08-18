/**
 *  @file Huffman.cuh
 *  @author Sheng Di
 *  Modified by Jiannan Tian
 *  @date Jan. 7, 2020
 *  Created on Aug., 2016
 *  @brief Customized Huffman Encoding, Compression and Decompression
 * functions. Also modified for GPU prototyping (header). (C) 2016 by
 * Mathematics and Computer Science (MCS), Argonne National Laboratory. See
 * COPYRIGHT in top-level directory.
 */

#ifndef A68D9FDE_BC7F_4E78_8532_93D427E3F126
#define A68D9FDE_BC7F_4E78_8532_93D427E3F126

#include <stdint.h>
#include <stdlib.h>

struct alignas(8) node_t {
  struct node_t *left, *right;
  size_t freq;
  char t;  // in_node:0; otherwise:1
  uint32_t c;
};

typedef struct node_t* node_list;

typedef struct alignas(8) hfserial_tree {
  uint32_t state_num;
  uint32_t all_nodes;
  struct node_t* pool;
  node_list *qqq, *qq;  // the root node of the hfserial_tree is qq[1]
  int n_nodes;          // n_nodes is for compression
  int qend;
  uint64_t** code;
  uint8_t* cout;
  int n_inode;  // n_inode is for decompression
} hfserial_tree;
typedef hfserial_tree HuffmanTree;

typedef struct alignas(8) HfSerialInternalStack {
  static const int MAX_DEPTH = 32;
  node_list _a[MAX_DEPTH];
  uint64_t saved_path[MAX_DEPTH];
  uint64_t saved_length[MAX_DEPTH];
  uint64_t depth = 0;
} hf_internal_stack;

static constexpr int HFSERIAL_TREE_CPU = 0;
static constexpr int HFSERIAL_TREE_CUDA = 1;

template <typename H>
void hf_buildtree_impl1(
    uint32_t* freq, uint16_t bklen, H* book, float* time = nullptr);

#endif /* A68D9FDE_BC7F_4E78_8532_93D427E3F126 */
