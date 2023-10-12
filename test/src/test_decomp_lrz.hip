// deps
#include "busyheader.hh"
#include "port.hh"
// definitions
#include "detail/t_scan.inl"

int main()
{
  auto all_pass = true, ok = true;

  ok = test_inclscan_1block<1, 256, 4>(), all_pass = all_pass and ok;
  printf("%s: test_inclscan_1block<1, 256, 4>()\n", ok ? "OK" : "WRONG");

  ok = test_inclscan_1block<1, 256, 8>(), all_pass = all_pass and ok;
  printf("%s: test_inclscan_1block<1, 256, 8>()\n", ok ? "OK" : "WRONG");

  ok = test_inclscan_1block<1, 512, 4>(), all_pass = all_pass and ok;
  printf("%s: test_inclscan_1block<1, 512, 4>()\n", ok ? "OK" : "WRONG");

  ok = test_inclscan_1block<1, 512, 8>(), all_pass = all_pass and ok;
  printf("%s: test_inclscan_1block<1, 512, 8>()\n", ok ? "OK" : "WRONG");

  ok = test_inclscan_1block<1, 1024, 4>(), all_pass = all_pass and ok;
  printf("%s: test_inclscan_1block<1, 1024, 4>()\n", ok ? "OK" : "WRONG");

  ok = test_inclscan_1block<1, 1024, 8>(), all_pass = all_pass and ok;
  printf("%s: test_inclscan_1block<1, 1024, 8>()\n", ok ? "OK" : "WRONG");

  ok = test_inclscan_1block<2>(), all_pass = all_pass and ok;
  printf("%s: test_inclscan_1block<2>()\n", ok ? "OK" : "WRONG");

  ok = test_inclscan_1block<3>(), all_pass = all_pass and ok;
  printf("%s: test_inclscan_1block<3>()\n", ok ? "OK" : "WRONG");

  if (all_pass)
    return 0;
  else
    return -1;
}