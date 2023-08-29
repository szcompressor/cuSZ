// deps
#include <fstream>

#include "busyheader.hh"
#include "port.hh"
// definitions
#include "detail/t_scan.inl"

int main()
{
  auto all_pass = true;
  all_pass = all_pass and test_inclusive_scan<256, 4>();
  all_pass = all_pass and test_inclusive_scan<256, 8>();
  all_pass = all_pass and test_inclusive_scan<512, 4>();
  all_pass = all_pass and test_inclusive_scan<512, 8>();
  all_pass = all_pass and test_inclusive_scan<1024, 4>();
  all_pass = all_pass and test_inclusive_scan<1024, 8>();

  if (all_pass)
    return 0;
  else
    return -1;
}