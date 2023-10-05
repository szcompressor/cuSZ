// deps
#include "busyheader.hh"
#include "port.hh"
#include <random>
// definitions
#include "detail/t_histsp.cu_hip.inl"

int main()
{
  constexpr auto large_radius = 64;
  constexpr auto NSYM = large_radius * 2;

  auto inlen = 500 * 500 * 100;
  auto all_eq = true;

  cudaDeviceReset();

  // all_eq = all_eq and test1_debug();
  all_eq = all_eq and test2_fulllen_input<NSYM>(inlen, dist3);
  // all_eq = all_eq and test3_performance_tuning<NSYM>(inlen, dist3);

  return all_eq ? 0 : -1;
}