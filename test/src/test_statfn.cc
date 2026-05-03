#include <cuda_runtime.h>

#include "compare.hh"
#include "cusz/type.h"
#include "rand.hh"
#include "utils/busyheader.hh"

void f(szt len, u4 seed)
{
  printf("len: %lu\n", len);

  f4 *d_in, *h_in;
  cudaMalloc(&d_in, len * sizeof(f4));
  cudaMallocHost(&h_in, len * sizeof(f4));

  psz::testutils::cu_hip::rand_array(d_in, len, seed);
  cudaMemcpy(h_in, d_in, len * sizeof(f4), cudaMemcpyDeviceToHost);

  f4 res_cpu[4], res_thrust[4], res_cuda[4];
  psz::analysis::probe_extrema<SEQ>(h_in, len, res_cpu);
#ifdef REACTIVATE_THRUSTGPU
  psz::probe_extrema<THRUST_DPL>(d_in, len, res_thrust);
#endif
  psz::analysis::probe_extrema<CUDA>(d_in, len, res_cuda);

  printf(
      "CPU\tmin: %6.4f\tmax: %6.4f\tavg: %6.4f\trng: %6.4f\n", res_cpu[0], res_cpu[1], res_cpu[2],
      res_cpu[3]);
#ifdef REACTIVATE_THRUSTGPU
  printf(
      "THRUST_DPL\tmin: %6.4f\tmax: %6.4f\tavg: %6.4f\trng: %6.4f\n", res_thrust[0], res_thrust[1],
      res_thrust[2], res_thrust[3]);
#endif
  printf(
      "CUDA\tmin: %6.4f\tmax: %6.4f\tavg: %6.4f\trng: %6.4f\n", res_cuda[0], res_cuda[1],
      res_cuda[2], res_cuda[3]);

  cudaFree(d_in);
  cudaFreeHost(h_in);
}

int main(int argc, char** argv)
{
  if (argc < 3)
    f(360 * 180, 0x246);
  else
    f(atoi(argv[1]), atoi(argv[2]));
  return 0;
}
