/**
 * @file hfbk_impl2.seq.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-08-17
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <queue>

#include "hfbk_impl.hh"
#include "hfword.hh"
#include "utils/timer.hh"

// impl2

// reference: https://gist.github.com/pwxcoo/72d7d3c5c3698371c21e486722f9b34b
template <typename H>
void phf_CPU_build_codebook_v2(u4* freq, size_t const bklen, H* book)
{
  using ref_node = phf_node*;
  using container = std::vector<ref_node>;

  std::priority_queue<phf_node*, container, phf_cmp_node> pq;

  for (auto i = 0; i < bklen; i++) {
    auto f = freq[i];
    if (f != 0) pq.push(new phf_node(i, f));
  }

  while (pq.size() != 1) {
    phf_node* left = pq.top();
    pq.pop();
    phf_node* right = pq.top();
    pq.pop();

    auto sum = left->freq + right->freq;
    pq.push(new phf_node(bklen + 1, sum, left, right));
  }
  phf_node* root = pq.top();
  phf_stack<phf_node, sizeof(H)>::template inorder_traverse<H>(root, book);
}

template void phf_CPU_build_codebook_v2(u4*, size_t const, u4*);
template void phf_CPU_build_codebook_v2(u4*, size_t const, u8*);
template void phf_CPU_build_codebook_v2(u4*, size_t const, ull*);

#undef NODE_STACK