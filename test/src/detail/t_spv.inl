/**
 * @file test_l3_spv.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-24
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "kernel/spv.hh"

template <typename T = float>
int f()
{
  T* a;                // input
  T* da;               // decoded
  size_t len = 10000;  //
  T* val;              // intermeidate
  uint32_t* idx;       //
  int nnz;             //
  float ms;

  GpuMallocManaged(&a, sizeof(T) * len);
  GpuMallocManaged(&da, sizeof(T) * len);
  GpuMallocManaged(&val, sizeof(T) * len);
  GpuMallocManaged(&idx, sizeof(uint32_t) * len);

  // determine nnz
  auto trials = randint(len) / 1;

  for (auto i = 0; i < trials; i++) {
    auto idx = randint(len);
    a[idx] = randint(INT32_MAX);
  }

  // CPU counting nnz
  auto nnz_ref = 0;
  for (auto i = 0; i < len; i++) {
    if (a[i] != 0) nnz_ref += 1;
  }

  GpuStreamT stream;
  GpuStreamCreate(&stream);

  ////////////////////////////////////////////////////////////////

  psz::spv_gather<PROPER_GPU_BACKEND, T, uint32_t>(a, len, val, idx, &nnz, &ms, stream);

  GpuStreamSync(stream);

  if (nnz != nnz_ref) {
    std::cout << "nnz_ref: " << nnz_ref << std::endl;
    std::cout << "nnz: " << nnz << std::endl;
    std::cerr << "nnz != nnz_ref" << std::endl;
    return -1;
  }

  psz::spv_scatter<PROPER_GPU_BACKEND, T, uint32_t>(val, idx, nnz, da, &ms, stream);

  GpuStreamSync(stream);

  ////////////////////////////////////////////////////////////////

  bool same = true;

  for (auto i = 0; i < len; i++) {
    if (a[i] != da[i]) {
      same = false;
      break;
    }
  }

  GpuFree(a);
  GpuFree(da);
  GpuFree(val);
  GpuFree(idx);

  GpuStreamDestroy(stream);

  if (same)
    return 0;
  else {
    std::cout << "decomp not okay" << std::endl;
    return -1;
  }
}
