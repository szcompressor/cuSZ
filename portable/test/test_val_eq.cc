#include <cassert>
#include <cstdio>

#include "cxx_type.hh"

using _portable::val_eq;

struct triple {
  int x, y, z;
};

int main()
{
  // Same values -> equal
  assert(val_eq(triple{1, 2, 3}, triple{1, 2, 3}));
  assert(val_eq(triple{0, 0, 0}, triple{0, 0, 0}));

  // Differs in any field -> not equal
  assert(not val_eq(triple{1, 2, 3}, triple{9, 2, 3}));
  assert(not val_eq(triple{1, 2, 3}, triple{1, 9, 3}));
  assert(not val_eq(triple{1, 2, 3}, triple{1, 2, 9}));

  // Works on _portable_len3 (the canonical xyz triple)
  _portable_len3 a{100, 200, 300};
  _portable_len3 b{100, 200, 300};
  _portable_len3 c{100, 200, 999};
  assert(val_eq(a, b));
  assert(not val_eq(a, c));

  printf("test_val_eq: PASS\n");
  return 0;
}
