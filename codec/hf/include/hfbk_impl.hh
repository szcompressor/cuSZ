/**
 * @file hfserial_book2.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-08-17
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef CD5DD212_2C45_4A8C_BDAD_7186A89BB353
#define CD5DD212_2C45_4A8C_BDAD_7186A89BB353

#include "cusz/type.h"
#include "hfword.hh"

// for impl1

struct alignas(8) node_t {
  struct node_t *left, *right;
  size_t freq;
  char t;  // in_node:0; otherwise:1
  union {
    uint32_t c;
    uint32_t symbol;
  };
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

template <typename H>
void phf_CPU_build_codebook_v1(u4* freq, uint16_t bklen, H* book);

// for impl2

struct phf_node {
  u4 symbol;
  u4 freq;
  phf_node *left, *right;

  phf_node(u4 symbol, u4 freq, phf_node* left = nullptr, phf_node* right = nullptr) :
      symbol(symbol), freq(freq), left(left), right(right)
  {
  }
};

struct phf_cmp_node {
  bool operator()(phf_node* left, phf_node* right) { return left->freq > right->freq; }
};

template <class NodeType, int WIDTH>
class alignas(8) phf_stack {
  static const int MAX_DEPTH = HuffmanWord<WIDTH>::FIELD_CODE;
  NodeType* _a[MAX_DEPTH];
  u8 saved_path[MAX_DEPTH];
  u8 saved_length[MAX_DEPTH];
  u8 depth = 0;

 public:
  static NodeType* top(phf_stack* s);

  template <typename T>
  static void push(phf_stack* s, NodeType* n, T path, T len);

  template <typename T>
  static NodeType* pop(phf_stack* s, T* path_to_restore, T* length_to_restore);

  template <typename H>
  static void inorder_traverse(NodeType* root, H* book);
};

template <typename H>
void phf_CPU_build_codebook_v2(u4* freq, size_t const bklen, H* book);

#endif /* CD5DD212_2C45_4A8C_BDAD_7186A89BB353 */
