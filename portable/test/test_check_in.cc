#include <cassert>
#include <cstdio>

#include "detail/check.hh"

using _ptb::detail::check_in;

int main()
{
  // Basic int membership
  assert(check_in(1, {1, 2, 3}, "", false));
  assert(check_in(2, {1, 2, 3}, "", false));
  assert(check_in(3, {1, 2, 3}, "", false));
  assert(not check_in(4, {1, 2, 3}, "", false));
  assert(not check_in(0, {1, 2, 3}, "", false));

  // Single-element list
  assert(check_in(42, {42}, "", false));
  assert(not check_in(43, {42}, "", false));

  // Negative values
  assert(check_in(-1, {-2, -1, 0}, "", false));
  assert(not check_in(1, {-2, -1, 0}, "", false));

  // throw_fail = false -> returns bool, no throw
  assert(not check_in(99, {1, 2, 3}, "should not throw", false));

  // Different integer type (size_t)
  assert(check_in<size_t>(10u, {5u, 10u, 15u}, "", false));

  printf("test_check_in: PASS\n");
  return 0;
}
