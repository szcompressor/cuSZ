// deps
#include "busyheader.hh"
#include "port.hh"
#include "rand.hh"
// definitions
#include "detail/t_cuda_pred.inl"

// TODO rename as cu_hip

int main(int argc, char** argv)
{
  bool all_pass = true;
  all_pass = all_pass and g(6480000, 1, 1);
  all_pass = all_pass and g(3600, 1800, 1);
  all_pass = all_pass and g(360, 180, 100);

  if (all_pass)
    return 0;
  else
    return -1;
}
