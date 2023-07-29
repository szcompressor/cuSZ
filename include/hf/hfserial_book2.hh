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

struct NodeCxx {
  u4 symbol;
  u4 freq;
  NodeCxx *left, *right;

  NodeCxx(
      u4 symbol, u4 freq, NodeCxx* left = nullptr, NodeCxx* right = nullptr) :
      symbol(symbol), freq(freq), left(left), right(right)
  {
  }
};

struct CmpNode {
  bool operator()(NodeCxx* left, NodeCxx* right)
  {
    return left->freq > right->freq;
  }
};

typedef struct alignas(8) __hf_stack {
  static const int MAX_DEPTH = 32;
  NodeCxx* _a[MAX_DEPTH];
  u8 saved_path[MAX_DEPTH];
  u8 saved_length[MAX_DEPTH];
  u8 depth = 0;

  static NodeCxx* top(__hf_stack* s);

  template <typename T>
  static void push(__hf_stack* s, NodeCxx* n, T path, T len);

  template <typename T>
  static NodeCxx* pop(__hf_stack* s, T* path_to_restore, T* length_to_restore);

  template <typename H>
  static void inorder_traverse(NodeCxx* root, H* book);
} __hf_stack;

template <typename H>
void hf_build_book2(u4* freq, size_t const bklen, H* book);

#endif /* CD5DD212_2C45_4A8C_BDAD_7186A89BB353 */
