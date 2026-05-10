#include <cassert>
#include <cstdio>

#include "detail/str2num.hh"

using _portable::detail::str_to_int;
using _portable::detail::str_to_num;

int main()
{
  // ── str_to_int: valid ────────────────────────────────────────────────
  {
    auto r = str_to_int("42");
    assert(r && *r == 42);
  }
  {
    auto r = str_to_int("-7");
    assert(r && *r == -7);
  }
  {
    auto r = str_to_int("0");
    assert(r && *r == 0);
  }

  // ── str_to_int: leading whitespace is accepted (strtoll convention) ──
  {
    auto r = str_to_int(" 42");
    assert(r && *r == 42);
  }

  // ── str_to_int: invalid ──────────────────────────────────────────────
  assert(not str_to_int(""));
  assert(not str_to_int("abc"));
  assert(not str_to_int("12abc"));    // trailing garbage
  assert(not str_to_int("42 "));      // trailing space (after int)
  assert(not str_to_int("3.14"));     // float, not int
  assert(not str_to_int(nullptr));

  // ── str_to_int: overflow ─────────────────────────────────────────────
  assert(not str_to_int("99999999999999999999999"));

  // ── str_to_num: valid ────────────────────────────────────────────────
  {
    auto r = str_to_num("3.14");
    assert(r && *r > 3.13 && *r < 3.15);
  }
  {
    auto r = str_to_num("1e-3");
    assert(r && *r > 9e-4 && *r < 1.1e-3);
  }
  {
    auto r = str_to_num("-2.5e2");
    assert(r && *r == -250.0);
  }
  {
    auto r = str_to_num("0");
    assert(r && *r == 0.0);
  }

  // ── str_to_num: invalid ──────────────────────────────────────────────
  assert(not str_to_num(""));
  assert(not str_to_num("abc"));
  assert(not str_to_num("1e"));       // incomplete exponent
  assert(not str_to_num(nullptr));

  printf("test_str2num: PASS\n");
  return 0;
}
