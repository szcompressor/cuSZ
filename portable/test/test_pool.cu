#include <cassert>
#include <cstdio>
#include <vector>

#include "mem/pool.h"

using _ptb::mem_plan;

namespace {

constexpr size_t ALIGN = mem_plan::_0x0100B;

enum : int { _TEST_EQ, _TEST_HIST, _TEST_ANCHOR, _TEST_S0, _TEST_D0 };

struct _test_buf : _ptb::buf_base {
  using _ptb::buf_base::buf_base;
  void init() override {}
  void reset(void*) override {}
};

void slices_are_contiguous_aligned_and_ordered()
{
  _test_buf a({{_TEST_EQ, U4, 300}}, {{_TEST_S0, U4, 1}});
  _test_buf b({{_TEST_HIST, U1, 7}}, {});
  _test_buf c({{_TEST_ANCHOR, U4, 64}}, {{_TEST_D0, U4, 5}});

  mem_plan::tree const t = _ptb::leaf(a) + _ptb::leaf(b) + _ptb::leaf(c);

  _ptb::pool pool;
  pool.allocate(t.bytes());
  t.assign({0, 0}, pool.data(), pool.state());

  auto const pa = (u1*)a.data().ptr_of<u4>(_TEST_EQ);
  auto const pb = (u1*)b.data().ptr_of<u1>(_TEST_HIST);
  auto const pc = (u1*)c.data().ptr_of<u4>(_TEST_ANCHOR);

  assert(pa == pool.data());
  assert(pb == (u1*)pool.data() + a.data().bytes());
  assert(pc == (u1*)pool.data() + a.data().bytes() + b.data().bytes());
  for (auto* p : {pa, pb, pc}) assert(((p - (u1*)pool.data()) % ALIGN) == 0);
  assert(pa + a.data().bytes() <= pb and pb + b.data().bytes() <= pc);
  assert(pc + c.data().bytes() <= (u1*)pool.data() + t.bytes().data);

  assert((u1*)a.state().ptr_of<u4>(_TEST_S0) == pool.state());
  assert((u1*)c.state().ptr_of<u4>(_TEST_D0) == (u1*)pool.state() + a.state().bytes());
}

void empty_pool_allocates_nothing()
{
  _ptb::pool pool;
  pool.allocate({0, 0});
  assert(pool.data() == nullptr and pool.state() == nullptr);
}

}  // namespace

int main()
{
  slices_are_contiguous_aligned_and_ordered();
  empty_pool_allocates_nothing();
  printf("test_pool: PASS\n");
  return 0;
}
