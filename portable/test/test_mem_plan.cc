#include <cassert>
#include <cstdio>
#include <vector>

#include "mem/plan.h"

using _ptb::mem_plan;


namespace {

constexpr size_t ALIGN = mem_plan::_0x0100B;

enum : int {  // test-local enum
  _TEST_EQ,
  _TEST_ANCHOR,
  _TEST_HIST,
  _TEST_D0,
  _TEST_D1,
  _TEST_S0,
  _TEST_USED,
  _TEST_UNUSED,
  _TEST_AFTER,
  _TEST_SCRATCH,
  _TEST_BITSTREAM,
  _TEST_PAR_NBIT,
  _TEST_TOTAL_NCELL,
  _TEST_INCOMP_FLAG,
  _TEST_ENCODED,
  _TEST_TREE_A,
  _TEST_TREE_B,
  _TEST_TREE_C
};

struct _test_buf : _ptb::buf_base {
  using _ptb::buf_base::buf_base;
  void init() override {}
  void reset(void*) override {}
};

bool aligned_and_disjoint(mem_plan::region const& b)
{
  auto const& n = b.v_nbyte();
  auto const& e = b.v_entry();

  for (size_t i = 0; i < b.n_frame(); i++) {
    if (not b.owns(i)) continue;
    if (e[i] % ALIGN != 0) return false;
    if (e[i] + n[i] > b.bytes()) return false;
  }

  // check for non-overlapping
  for (size_t i = 0; i < b.n_frame(); i++)
    for (size_t j = i + 1; j < b.n_frame(); j++) {
      if (not b.owns(i) or not b.owns(j)) continue;
      if (n[i] == 0 or n[j] == 0) continue;
      if (e[i] < e[j] + n[j] and e[j] < e[i] + n[i]) return false;
    }

  return true;
}

mem_plan hf_plan(size_t len, size_t pardeg)
{
  return mem_plan(
      {{_TEST_SCRATCH, U4, len}, {_TEST_BITSTREAM, U4, len / 2}, {_TEST_PAR_NBIT, U4, pardeg}},
      {{_TEST_TOTAL_NCELL, U4, 1}, {_TEST_INCOMP_FLAG, U1, pardeg}});
}

}  // namespace

void len_is_element_count()
{
  mem_plan const p({{_TEST_EQ, U2, 1000}, {_TEST_ANCHOR, F4, 100}}, {{_TEST_HIST, U4, 1024}});

  assert(p.data().nbyte(p.data().find(_TEST_EQ)) == 2000);
  assert(p.data().nbyte(p.data().find(_TEST_ANCHOR)) == 400);
  assert(p.state().nbyte(p.state().find(_TEST_HIST)) == 4096);
  assert(aligned_and_disjoint(p.data()) and aligned_and_disjoint(p.state()));
}

void sym_vs_index()
{
  mem_plan const p({{_TEST_ANCHOR, F4, 100}, {_TEST_EQ, U2, 1000}});
  assert(p.data().find(_TEST_EQ) == 1 and p.data().find(_TEST_ANCHOR) == 0);
  assert(p.data().find(_TEST_HIST) == mem_plan::npos);
  assert(p.data().sym(0) == _TEST_ANCHOR);
}

void regions_independent()
{
  mem_plan const p({{_TEST_D0, U1, 1}, {_TEST_D1, U1, 1}}, {{_TEST_S0, U1, 1}});

  assert(p.data().entry(0) == 0 and p.data().entry(1) == ALIGN);
  assert(p.state().entry(0) == 0);
  assert(p.data().bytes() == 2 * ALIGN and p.state().bytes() == ALIGN);
  assert(p.data().n_frame() == 2 and p.state().n_frame() == 1);
}

void zero_len_is_null()
{
  mem_plan p({{_TEST_USED, U4, 64}, {_TEST_UNUSED, U4, 0}, {_TEST_AFTER, U4, 64}});

  assert(p.data().bytes() == 2 * ALIGN);
  assert(p.data().entry(1) == p.data().entry(2));

  std::vector<u1> data(p.data().bytes());
  p.set_base(data.data(), nullptr);
  assert(p.data().ptr<u4>(1) == nullptr);
  assert(p.data().ptr<u4>(0) == (u4*)data.data());
}

void ptr_matches_entry()
{
  auto            p = hf_plan(256, 4);
  std::vector<u1> data(p.data().bytes()), state(p.state().bytes());

  p.set_base(data.data(), state.data());
  for (size_t i = 0; i < p.data().n_frame(); i++)
    assert(p.data().ptr<u1>(i) == data.data() + p.data().entry(i));
  for (size_t i = 0; i < p.state().n_frame(); i++)
    assert(p.state().ptr<u1>(i) == state.data() + p.state().entry(i));

  assert(aligned_and_disjoint(p.data()) and aligned_and_disjoint(p.state()));
}

void reinterpret_width()
{
  mem_plan        p({{_TEST_EQ, U4, 8}});
  std::vector<u1> data(p.data().bytes());
  p.set_base(data.data(), nullptr);

  auto* as_u4 = p.data().ptr<u4>(0);
  auto* as_u2 = p.data().ptr<u2>(0);
  as_u4[0]    = 0x00010001u;
  assert(as_u2[0] == 1 and as_u2[1] == 1);
}

void dup_sym()
{
  mem_plan const dup({{_TEST_EQ, U4, 8}, {_TEST_EQ, U2, 8}});
  assert(dup.status() == _FAIL_GENERAL);

  // the same symbol in the other region is fine: lookup is per region
  mem_plan const shared({{_TEST_EQ, U4, 8}}, {{_TEST_EQ, U2, 8}});
  assert(shared.status() == _SUCCESS);
}

void phf_alias()
{
  size_t const words = 1024;
  mem_plan     p({{_TEST_SCRATCH, U4, words}, {_TEST_ENCODED, U1, 0, _TEST_SCRATCH}});

  assert(p.status() == _SUCCESS);
  assert(p.data().owns(0) and not p.data().owns(1));
  assert(p.data().entry(1) == p.data().entry(0));
  assert(p.data().nbyte(1) == words * 4);  // len 0 took the whole target
  assert(p.data().bytes() == words * 4);   // the view added nothing

  std::vector<u1> data(p.data().bytes());
  p.set_base(data.data(), nullptr);
  assert((void*)p.data().ptr<u1>(1) == (void*)p.data().ptr<u4>(0));
  assert(aligned_and_disjoint(p.data()));
}

void bad_alias()
{
  mem_plan const too_big({{_TEST_SCRATCH, U4, 8}, {_TEST_ENCODED, U1, 33, _TEST_SCRATCH}});
  assert(too_big.status() == _FAIL_GENERAL);

  mem_plan const forward({{_TEST_ENCODED, U1, 8, _TEST_SCRATCH}, {_TEST_SCRATCH, U4, 8}});
  assert(forward.status() == _FAIL_GENERAL);

  mem_plan const unknown({{_TEST_SCRATCH, U4, 8}, {_TEST_ENCODED, U1, 8, _TEST_HIST}});
  assert(unknown.status() == _FAIL_GENERAL);

  mem_plan const chained({{_TEST_SCRATCH, U4, 8},
                          {_TEST_ENCODED, U1, 0, _TEST_SCRATCH},
                          {_TEST_EQ, U1, 0, _TEST_ENCODED}});
  assert(chained.status() == _FAIL_GENERAL);  // a view of a view is rejected
}

void absent_sym_is_null()
{
  mem_plan p({{_TEST_EQ, U4, 8}, {_TEST_HIST, U4, 4}}, {{_TEST_S0, U4, 2}});
  std::vector<u1> d(p.data().bytes()), s(p.state().bytes());
  p.set_base(d.data(), s.data());

  assert(p.data().find(_TEST_EQ) == 0);
  assert(p.data().find(_TEST_HIST) == 1);
  assert(p.data().find(_TEST_ANCHOR) == mem_plan::npos);

  assert(p.data().ptr_of<u4>(_TEST_EQ) == (u4*)d.data());
  assert(p.state().ptr_of<u4>(_TEST_S0) == (u4*)s.data());
  assert(p.data().ptr_of<u4>(_TEST_ANCHOR) == nullptr);
  assert(p.data().ptr_of<u4>(_TEST_S0) == nullptr);
  assert(p.state().ptr_of<u4>(_TEST_EQ) == nullptr);
}

void no_base_means_null()
{
  mem_plan const p({{_TEST_EQ, U4, 8}});
  assert(p.data().ptr<u4>(0) == nullptr);
}

void plan_totals()
{
  mem_plan lrz({{_TEST_EQ, U4, 64}});                                 // data 256 B
  mem_plan spl({{_TEST_EQ, U4, 64}, {_TEST_ANCHOR, U4, 64}});         // data 512 B
  mem_plan out({{_TEST_D0, U4, 64}}, {{_TEST_S0, U4, 1}});            // 256 B / 256 B

  mem_plan::total const pred = lrz | spl;
  assert(pred.data == 2 * ALIGN);                                     // max, not sum

  mem_plan::total const all = (lrz | spl) + out;
  assert(all.data == 3 * ALIGN);
  assert(all.state == ALIGN);
}

void plan_laws()
{
  mem_plan a({{_TEST_EQ, U4, 64}}), b({{_TEST_ANCHOR, U4, 128}}), c({{_TEST_HIST, U4, 32}});

  assert(((a + b) + c).data == (a + (b + c)).data);
  assert(((a | b) | c).data == (a | (b | c)).data);
  assert((a | b).data == (b | a).data);
}

void plan_bases()
{
  mem_plan lrz({{_TEST_EQ, U4, 64}});
  mem_plan spl({{_TEST_ANCHOR, U4, 64}});
  mem_plan out({{_TEST_D0, U4, 64}});

  mem_plan::total const all = (lrz | spl) + out;
  std::vector<u1> d(all.data), s(all.state + 1);

  // the caller places them: alternatives share a base, what follows starts past the wider
  auto const alt = (mem_plan::total)lrz | (mem_plan::total)spl;
  lrz.set_base(d.data(), s.data());
  spl.set_base(d.data(), s.data());
  out.set_base(d.data() + alt.data, s.data() + alt.state);

  assert(lrz.data().ptr_of<u1>(_TEST_EQ) == d.data());
  assert(spl.data().ptr_of<u1>(_TEST_ANCHOR) == d.data());
  assert(out.data().ptr_of<u1>(_TEST_D0) == d.data() + ALIGN);
}

void tree_add_is_partial_sum()
{
  mem_plan::total const x{100, 0}, y{200, 0}, z{50, 0};
  mem_plan::tree const t = mem_plan::tree(x) + y + z;

  assert(t.bytes().data == 350);
  auto const off = t.assign();
  assert(off.size() == 3);
  assert(off[0].data == 0);
  assert(off[1].data == 100);
  assert(off[2].data == 300);
}

void tree_alt_shares_base()
{
  mem_plan::total const x{100, 0}, y{300, 0}, z{50, 0};
  mem_plan::tree const t = mem_plan::tree(x) | y | z;

  assert(t.bytes().data == 300);
  auto const off = t.assign();
  assert(off.size() == 3);
  assert(off[0].data == 0 and off[1].data == 0 and off[2].data == 0);
}

void tree_offsets_track_both_regions()
{
  mem_plan::total const a{10, 1}, b{20, 2};
  mem_plan::tree const t = mem_plan::tree(a) + b;

  auto const off = t.assign();
  assert(off[0].data == 0 and off[0].state == 0);
  assert(off[1].data == 10 and off[1].state == 1);
}

void tree_attaches_leaves()
{
  _test_buf a({{_TEST_TREE_A, U4, 64}}, {{_TEST_S0, U4, 1}});
  _test_buf b({{_TEST_TREE_B, U1, 7}}, {});
  _test_buf c({{_TEST_TREE_C, U4, 8}}, {{_TEST_D0, U4, 5}});

  mem_plan::tree const t = _ptb::leaf(a) + _ptb::leaf(b) + _ptb::leaf(c);
  auto const bytes = t.bytes();

  std::vector<u1> d(bytes.data), s(bytes.state);
  t.assign({0, 0}, d.data(), s.data());

  assert((u1*)a.data().ptr_of<u4>(_TEST_TREE_A) == d.data());
  assert((u1*)b.data().ptr_of<u1>(_TEST_TREE_B) == d.data() + a.data().bytes());
  assert((u1*)c.data().ptr_of<u4>(_TEST_TREE_C) == d.data() + a.data().bytes() + b.data().bytes());
  assert((u1*)a.state().ptr_of<u4>(_TEST_S0) == s.data());
  assert((u1*)c.state().ptr_of<u4>(_TEST_D0) == s.data() + a.state().bytes());
}

void tree_attaches_when_one_region_is_empty()
{
  _test_buf state_only({}, {{_TEST_S0, U4, 1}});
  mem_plan::tree const t = _ptb::leaf(state_only);

  assert(t.bytes().data == 0);
  std::vector<u1> s(t.bytes().state);
  t.assign({0, 0}, nullptr, s.data());

  assert((u1*)state_only.state().ptr_of<u4>(_TEST_S0) == s.data());
}

void tree_matches_stage1_shape()
{
  mem_plan::total const lrz{100, 0}, y24{120, 0}, y25{150, 0}, anchor{20, 0}, outlier{40, 0};

  mem_plan::tree const stage1 = (lrz | ((mem_plan::tree(y24) | y25) + anchor)) + outlier;

  assert(stage1.bytes().data == 210);

  auto const off = stage1.assign();
  assert(off.size() == 5);
  assert(off[0].data == 0);
  assert(off[1].data == 0);
  assert(off[2].data == 0);
  assert(off[3].data == 150);
  assert(off[4].data == 170);
}

int main()
{
  len_is_element_count();
  sym_vs_index();
  regions_independent();
  zero_len_is_null();
  ptr_matches_entry();
  reinterpret_width();
  dup_sym();
  phf_alias();
  bad_alias();
  absent_sym_is_null();
  no_base_means_null();
  plan_totals();
  plan_laws();
  plan_bases();
  tree_add_is_partial_sum();
  tree_alt_shares_base();
  tree_offsets_track_both_regions();
  tree_attaches_leaves();
  tree_attaches_when_one_region_is_empty();
  tree_matches_stage1_shape();
  printf("test_mem_plan: PASS\n");
  return 0;
}
