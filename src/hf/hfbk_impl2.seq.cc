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

#include "busyheader.hh"
#include "hf/hfbk_impl.hh"
#include "hf/hfword.hh"
#include "utils/timer.hh"

// impl2

// reference: https://gist.github.com/pwxcoo/72d7d3c5c3698371c21e486722f9b34b
template <typename H>
void hf_buildtree_impl2(u4* freq, size_t const bklen, H* book, float* time)
{
  using NodeType = NodeCxx;
  using Container = std::vector<NodeCxx*>;

  std::priority_queue<NodeType*, Container, CmpNode> pq;

  auto a = hires::now();

  for (auto i = 0; i < bklen; i++) {
    auto f = freq[i];
    if (f != 0) pq.push(new NodeType(i, f));
  }

  while (pq.size() != 1) {
    NodeType* left = pq.top();
    pq.pop();
    NodeType* right = pq.top();
    pq.pop();

    auto sum = left->freq + right->freq;
    pq.push(new NodeType(bklen + 1, sum, left, right));
  }

  NodeType* root = pq.top();

  __pszhf_stack<NodeType, sizeof(H)>::template inorder_traverse<H>(root, book);

  auto b = hires::now();
  auto t = static_cast<duration_t>(b - a).count() * 1000;
  if (time) *time = t;
}

template void hf_buildtree_impl2(u4*, size_t const, u4*, f4*);
template void hf_buildtree_impl2(u4*, size_t const, u8*, f4*);
template void hf_buildtree_impl2(u4*, size_t const, ull*, f4*);

#undef NodeStack