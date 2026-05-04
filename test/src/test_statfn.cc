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

  auto [cpu_min, cpu_max, cpu_avg, cpu_rng] = psz::analysis::CPU_probe_extrema<f4, SEQ>(h_in, len);
#ifdef REACTIVATE_THRUSTGPU
  auto [thrust_min, thrust_max, thrust_avg, thrust_rng] =
      psz::GPU_probe_extrema<f4, THRUST_DPL>(d_in, len);
#endif
  auto [cuda_min, cuda_max, cuda_avg, cuda_rng] =
      psz::analysis::GPU_probe_extrema<f4, CUDA>(d_in, len);

  printf(
      "CPU\tmin: %6.4f\tmax: %6.4f\tavg: %6.4f\trng: %6.4f\n", cpu_min, cpu_max, cpu_avg, cpu_rng);
#ifdef REACTIVATE_THRUSTGPU
  printf(
      "THRUST_DPL\tmin: %6.4f\tmax: %6.4f\tavg: %6.4f\trng: %6.4f\n", thrust_min, thrust_max,
      thrust_avg, thrust_rng);
#endif
  printf(
      "CUDA\tmin: %6.4f\tmax: %6.4f\tavg: %6.4f\trng: %6.4f\n", cuda_min, cuda_max, cuda_avg,
      cuda_rng);

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
