// deps
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "busyheader.hh"
#include "port.hh"
#include "rand.hh"
#include <random>
// definitions
#include "detail/t_lrzsp.dp.inl"

bool batch_run_testcase(uint32_t x, uint32_t y, uint32_t z)
{
  auto all_pass = true;

  auto ndim = [&]() -> int {
    auto _ndim = 3;
    if (z == 1) _ndim = 2;
    if (y == 1) _ndim = 1;
    if (x == 1) throw std::runtime_error("x cannot be 1");
    return _ndim;
  };

  double eb = 1e-3;  // for 1D
  if (ndim() == 2) eb = 3e-3;
  if (ndim() == 3) eb = 3e-3;

  // all_pass = all_pass and testcase<float, uint8_t>(x, y, z, eb, 128);
  // all_pass = all_pass and testcase<float, uint16_t>(x, y, z, eb, 512);
  all_pass = all_pass and testcase<float, uint32_t>(x, y, z, eb, 512);
  // all_pass = all_pass and testcase<float, float>(x, y, z, eb, 512);
  // all_pass = all_pass and testcase<double, uint8_t>(x, y, z, eb, 128);
  // all_pass = all_pass and testcase<double, uint16_t>(x, y, z, eb, 512);
  // all_pass = all_pass and testcase<double, uint32_t>(x, y, z, eb, 512);
  // all_pass = all_pass and testcase<double, float>(x, y, z, eb, 512);

  // all_pass = all_pass and testcase<float, int32_t>(x, y, z, eb, 512);
  // all_pass = all_pass and testcase<double, int32_t>(x, y, z, eb, 512);

  return all_pass;
}

int main(int argc, char** argv)
{
  dpct::get_current_device().reset();

  bool all_pass = true;

  all_pass = all_pass and batch_run_testcase(6480000, 1, 1);
  printf("\n");
  all_pass = all_pass and batch_run_testcase(3600, 1800, 1);
  printf("\n");
  all_pass = all_pass and batch_run_testcase(360, 180, 100);

  if (all_pass)
    return 0;
  else
    return -1;
}
