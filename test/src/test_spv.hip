// deps
#include <ctime>
#include <random>
#include "busyheader.hh"
#include "port.hh"
#include "rand.hh"
// definitions
#include "detail/t_spv.cu_hip.inl"

int main()
{
  auto all_pass = true;
  auto pass = true;
  for (auto i = 0; i < 10; i++) {
    pass = f() == 0;
    if (not pass) printf("Not passed on %dth trial.\n", i);
    all_pass &= pass;
  }

  if (all_pass)
    return 0;
  else
    return -1;
}