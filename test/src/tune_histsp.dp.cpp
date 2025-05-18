// deps
#include <dpct/dpct.hpp>
#include <random>
#include <sycl/sycl.hpp>

#include "detail/busyheader.hh"
// definitions
#include "detail/t_histsp.dp.inl"

int main()
{
  constexpr auto large_radius = 64;
  constexpr auto NSYM = large_radius * 2;

  auto inlen = 500 * 500 * 100;
  auto all_eq = true;

  dpct::get_current_device().reset();

  // all_eq = all_eq and test1_debug();
  all_eq = all_eq and test2_fulllen_input<NSYM>(inlen, dist3);
  // all_eq = all_eq and test3_performance_tuning<NSYM>(inlen, dist3);

  return all_eq ? 0 : -1;
}