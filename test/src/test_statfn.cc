/**
 * @file test_statfn.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-08-20
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <algorithm>

#include "busyheader.hh"
#include "cusz/type.h"
#include "mem/memseg.h"
#include "mem/memseg_cxx.hh"
#include "rand.hh"
#include "stat/compare.hh"

void f(szt len, u4 seed)
{
  printf("len: %lu\n", len);

  auto in_cpu = new pszmem_cxx<f4>(len, 1, 1, "f4 cpu");
  auto in_thrust = new pszmem_cxx<f4>(len, 1, 1, "f4 thrust");
  auto in_cuda = new pszmem_cxx<f4>(len, 1, 1, "f4 cu");

  in_cpu->control({Malloc, MallocHost});
  in_thrust->control({Malloc, MallocHost});
  in_cuda->control({Malloc, MallocHost});

  psz::testutils::cu_hip::rand_array(in_cpu->dptr(), in_cpu->len(), seed);

  in_cpu->control({D2H});

#if defined(PSZ_USE_CUDA)
  pszmem_device_deepcopy_cuda(in_cuda->m, in_cpu->m);
  pszmem_device_deepcopy_cuda(in_thrust->m, in_cpu->m);
#elif defined(PSZ_USE_HIP)
  pszmem_device_deepcopy_hip(in_cuda->m, in_cpu->m);
  pszmem_device_deepcopy_hip(in_thrust->m, in_cpu->m);
#endif

  f4 res_cpu[4], res_thrust[4], res_cuda[4];
  psz::probe_extrema<SEQ>(in_cpu->hptr(), len, res_cpu);
#ifdef REACTIVATE_THRUSTGPU
  psz::probe_extrema<THRUST>(in_thrust->dptr(), len, res_thrust);
#endif
  // In fact, below is CUDA-HIP compat. Need better indication.
  psz::probe_extrema<CUDA>(in_cuda->dptr(), len, res_cuda);

  printf(
      "CPU\tmin: %6.4f\tmax: %6.4f\tavg: %6.4f\trng: %6.4f\n",  //
      res_cpu[0], res_cpu[1], res_cpu[2], res_cpu[3]);
#ifdef REACTIVATE_THRUSTGPU
  printf(
      "THRUST\tmin: %6.4f\tmax: %6.4f\tavg: %6.4f\trng: %6.4f\n",  //
      res_thrust[0], res_thrust[1], res_thrust[2], res_thrust[3]);
#endif
  printf(
      "CUDA\tmin: %6.4f\tmax: %6.4f\tavg: %6.4f\trng: %6.4f\n",  //
      res_cuda[0], res_cuda[1], res_cuda[2], res_cuda[3]);
}

int main(int argc, char** argv)
{
  if (argc < 3)
    f(360 * 180, 0x246);
  else
    f(atoi(argv[1]), atoi(argv[2]));
  return 0;
}