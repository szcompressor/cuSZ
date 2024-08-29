// deps
#include <fstream>

#include "busyheader.hh"
#include "port.hh"
// definitions
// #include "detail/t_cudaproto.inl"


// TODO use low-level API instead of launch kernel

int main()
{
  auto all_pass = true;

  // all_pass = all_pass and run_test1();
  // all_pass = all_pass and run_test2();
  // all_pass = all_pass and run_test3();

  if (all_pass)
    return 0;
  else
    return -1;
}
