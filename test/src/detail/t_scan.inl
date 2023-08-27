/**
 * @file test_l1_l23scan.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2022-12-23
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "kernel/detail/l23.inl"
#include "../rand.hh"

template <int BLOCK = 256, int SEQ = 8>
bool test_inclusive_scan()
{
  using T = float;
  using EQ = int32_t;
  using FP = T;

  constexpr auto NTHREAD = BLOCK / SEQ;

  auto len = BLOCK;
  auto ebx2 = 1;

  T* data{nullptr};
  EQ* eq{nullptr};

  GpuMallocManaged(&data, sizeof(T) * len);
  GpuMallocManaged(&eq, sizeof(EQ) * len);
  GpuMemset(eq, 0x0, sizeof(EQ) * len);

  bool ok = true;

  {
    cout << "refactored v0 (wave32)" << endl;
    for (auto i = 0; i < BLOCK; i++) data[i] = 1;

    psz::cuda_hip::__kernel::v0::x_lorenzo_1d1l<T, EQ, FP, BLOCK, SEQ>
        <<<1, NTHREAD>>>(
            eq, data, dim3(len, 1, 1), dim3(0, 0, 0), 0, ebx2, data);
    GpuDeviceSync();

    // for (auto i = 0; i < BLOCK; i++) cout << data[i] << " ";
    // cout << "\n" << endl;

    for (auto i = 0; i < BLOCK; i++)
      if (data[i] != i + 1 /* inclusive scan */) {
        ok = false;
        break;
      }

    if (not ok) {
      printf("ERROR\trefactored (22-Dec) not correct\n");
      return false;
    }
  }

  GpuFree(data);
  GpuFree(eq);

  return ok;
}