/**
 *  @file hfbk_impl1.seq.cc
 *  @author Sheng Di
 *  Modified by Jiannan Tian
 *  @date Jan. 7, 2020
 *  Created on Aug., 2016
 *  @brief Customized Huffman Encoding, Compression and Decompression
 * functions. Also modified for GPU prototyping. (C) 2016 by Mathematics and
 * Computer Science (MCS), Argonne National Laboratory. See COPYRIGHT in
 * top-level directory.
 */


#include "busyheader.hh"
#include "cusz/type.h"
#include "hf/hfbk_impl.hh"
#include "hf/hfword.hh"
#include "utils/timer.hh"

HuffmanTree* create_tree_serial(int state_num)
{
  auto ht = (HuffmanTree*)malloc(sizeof(HuffmanTree));
  memset(ht, 0, sizeof(HuffmanTree));
  ht->state_num = state_num;
  ht->all_nodes = 2 * state_num;

  ht->pool = (struct node_t*)malloc(ht->all_nodes * 2 * sizeof(struct node_t));
  ht->qqq = (node_list*)malloc(ht->all_nodes * 2 * sizeof(node_list));
  ht->code = (uint64_t**)malloc(ht->state_num * sizeof(uint64_t*));
  ht->cout = (uint8_t*)malloc(ht->state_num * sizeof(uint8_t));

  memset(ht->pool, 0, ht->all_nodes * 2 * sizeof(struct node_t));
  memset(ht->qqq, 0, ht->all_nodes * 2 * sizeof(node_list));
  memset(ht->code, 0, ht->state_num * sizeof(uint64_t*));
  memset(ht->cout, 0, ht->state_num * sizeof(uint8_t));
  ht->qq = ht->qqq - 1;
  ht->n_nodes = 0;
  ht->n_inode = 0;
  ht->qend = 1;

  return ht;
}

void destroy_tree(HuffmanTree* ht)
{
  size_t i;
  free(ht->pool);
  ht->pool = nullptr;
  free(ht->qqq);
  ht->qqq = nullptr;
  for (i = 0; i < ht->state_num; i++) {
    if (ht->code[i] != nullptr) free(ht->code[i]);
  }
  free(ht->code);
  ht->code = nullptr;
  free(ht->cout);
  ht->cout = nullptr;
  free(ht);
  ht = nullptr;
}

HuffmanTree* create_hfserial_tree_gpu(int state_num)
{
  auto ht = (HuffmanTree*)malloc(sizeof(HuffmanTree));
  memset(ht, 0, sizeof(HuffmanTree));
  ht->state_num = state_num;
  ht->all_nodes = 2 * state_num;

  ht->pool = (struct node_t*)malloc(ht->all_nodes * 2 * sizeof(struct node_t));
  ht->qqq = (node_list*)malloc(ht->all_nodes * 2 * sizeof(node_list));
  ht->code = (uint64_t**)malloc(ht->state_num * sizeof(uint64_t*));
  ht->cout = (uint8_t*)malloc(ht->state_num * sizeof(uint8_t));

  memset(ht->pool, 0, ht->all_nodes * 2 * sizeof(struct node_t));
  memset(ht->qqq, 0, ht->all_nodes * 2 * sizeof(node_list));
  memset(ht->code, 0, ht->state_num * sizeof(uint64_t*));
  memset(ht->cout, 0, ht->state_num * sizeof(uint8_t));
  ht->qq = ht->qqq - 1;
  ht->n_nodes = 0;
  ht->n_inode = 0;
  ht->qend = 1;

  return ht;
}

node_list new_node(
    HuffmanTree* ht, size_t freq, uint32_t c, node_list a, node_list b)
{
  node_list n = ht->pool + ht->n_nodes++;
  if (freq) {
    n->c = c;
    n->freq = freq;
    n->t = 1;
  }
  else {
    n->left = a;
    n->right = b;
    n->freq = a->freq + b->freq;
    n->t = 0;
    // n->c = 0;
  }
  return n;
}

/* priority queue */
void qinsert(HuffmanTree* ht, node_list n)
{
  int j, i = ht->qend++;
  while ((j = (i >> 1))) {  // j=i/2
    if (ht->qq[j]->freq <= n->freq) break;
    ht->qq[i] = ht->qq[j], i = j;
  }
  ht->qq[i] = n;
}

node_list qremove(HuffmanTree* ht)
{
  int i, l;
  node_list n = ht->qq[i = 1];
  node_list p;
  if (ht->qend < 2) return 0;
  ht->qend--;
  ht->qq[i] = ht->qq[ht->qend];

  while ((l = (i << 1)) < ht->qend)  // l=(i*2)
  {
    if (l + 1 < ht->qend && ht->qq[l + 1]->freq < ht->qq[l]->freq) l++;
    if (ht->qq[i]->freq > ht->qq[l]->freq) {
      p = ht->qq[i];
      ht->qq[i] = ht->qq[l];
      ht->qq[l] = p;
      i = l;
    }
    else {
      break;
    }
  }

  return n;
}

/* walk the tree and put 0s and 1s */
/**
 * @out1 should be set to 0.
 * @out2 should be 0 as well.
 * @index: the index of the byte
 * */
void build_code(
    HuffmanTree* ht, node_list n, int len, uint64_t out1, uint64_t out2)
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

template <typename H>
void hf_buildtree_impl1(
    uint32_t* ext_freq, uint16_t booklen, H* book, float* time)
{
  using NodeType = node_t;

  auto state_num = 2 * booklen;
  auto all_nodes = 2 * state_num;

  auto freq = new uint32_t[all_nodes];
  memset(freq, 0, sizeof(uint32_t) * all_nodes);
  memcpy(freq, ext_freq, sizeof(uint32_t) * booklen);

  auto tree = create_tree_serial(state_num);

  {  // real "kernel"
    auto a = hires::now();

    for (size_t i = 0; i < tree->all_nodes; i++)
      if (freq[i]) qinsert(tree, new_node(tree, freq[i], i, 0, 0));
    while (tree->qend > 2)
      qinsert(tree, new_node(tree, 0, 0, qremove(tree), qremove(tree)));
    __pszhf_stack<NodeType, sizeof(H)>::template inorder_traverse<H>(tree->qq[1], book);

    auto b = hires::now();
    auto t = static_cast<duration_t>(b - a).count() * 1000;
    if (time) *time = t;
  }

  destroy_tree(tree);
  delete[] freq;
}

template void hf_buildtree_impl1<u4>(u4*, u2, u4*, f4*);
template void hf_buildtree_impl1<u8>(u4*, u2, u8*, f4*);
template void hf_buildtree_impl1<ull>(u4*, u2, ull*, f4*);
