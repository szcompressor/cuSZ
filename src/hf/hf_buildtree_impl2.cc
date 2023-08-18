/**
 * @file hfserial_book2.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-08-17
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include "hf/hf_buildtree_impl2.hh"

#include <queue>

#include "busyheader.hh"
#include "utils/timer.hh"

NodeCxx* __hf_stack::top(__hf_stack* s) { return s->_a[s->depth - 1]; }

template <typename T>
void __hf_stack::push(__hf_stack* s, NodeCxx* n, T path, T len)
{
  if (s->depth + 1 <= __hf_stack::MAX_DEPTH) {
    s->depth += 1;

    s->_a[s->depth - 1] = n;
    s->saved_path[s->depth - 1] = path;
    s->saved_length[s->depth - 1] = len;
  }
  else
    printf("Error: stack overflow\n");
}

template <typename T>
NodeCxx* __hf_stack::pop(
    __hf_stack* s, T* path_to_restore, T* length_to_restore)
{
  auto is_empty = [&](__hf_stack* s) -> bool { return (s->depth == 0); };

  NodeCxx* n;

  if (is_empty(s)) {
    printf("Error: stack underflow, exiting...\n");
    return nullptr;
  }
  else {
    // TODO holding array -> __a
    n = s->_a[s->depth - 1];
    s->_a[s->depth - 1] = nullptr;

    *length_to_restore = s->saved_length[s->depth - 1];
    *path_to_restore = s->saved_path[s->depth - 1];
    s->depth -= 1;

    return n;
  }
}

template <typename H>
void __hf_stack::inorder_traverse(NodeCxx* root, H* book)
{
  auto is_empty = [&](__hf_stack* s) -> bool { return (s->depth == 0); };

  auto s = new __hf_stack();

  bool done = 0;
  H out1 = 0, len = 0;

  while (not done) {
    if (root->left or root->right) {
      push(s, root, out1, len);
      root = root->left;
      out1 <<= 1u;
      out1 |= 0u;
      len += 1;
    }
    else {
      u4 symbol = root->symbol;
      book[symbol] = out1 | ((len & (H)0xffu) << (sizeof(H) * 8 - 8));
      if (!is_empty(s)) {
        root = pop(s, &out1, &len);
        root = root->right;
        out1 <<= 1u;
        out1 |= 1u;
        len += 1;
      }
      else
        done = true;
    }
  }

  delete s;
  /* end of function */
}

// reference: https://gist.github.com/pwxcoo/72d7d3c5c3698371c21e486722f9b34b
template <typename H>
void hf_buildtree_impl2(u4* freq, size_t const bklen, H* book, float* time)
{
  using N = NodeCxx;
  using Container = std::vector<NodeCxx*>;

  std::priority_queue<N*, Container, CmpNode> pq;

  auto a = hires::now();

  for (auto i = 0; i < bklen; i++) {
    auto f = freq[i];
    if (f != 0) pq.push(new N(i, f));
  }

  while (pq.size() != 1) {
    N* left = pq.top();
    pq.pop();
    N* right = pq.top();
    pq.pop();

    auto sum = left->freq + right->freq;
    pq.push(new N(bklen + 1, sum, left, right));
  }

  N* root = pq.top();

  __hf_stack::inorder_traverse(root, book);

  auto b = hires::now();
  auto t = static_cast<duration_t>(b - a).count() * 1000;
  if (time) *time = t;
}

template void hf_buildtree_impl2(u4*, size_t const, u4*, f4*);
template void hf_buildtree_impl2(u4*, size_t const, u8*, f4*);
template void hf_buildtree_impl2(u4*, size_t const, ull*, f4*);