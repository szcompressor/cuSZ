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

#include "kernel/criteria.gpu.hh"
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
  int* d_nnz;
  float ms;

  cudaMallocManaged(&a, sizeof(T) * len);
  cudaMallocManaged(&da, sizeof(T) * len);
  cudaMallocManaged(&val, sizeof(T) * len);
  cudaMallocManaged(&idx, sizeof(uint32_t) * len);

  cudaMallocManaged(&d_nnz, sizeof(int));

  // determine nnz
  auto trials = psz::testutils::cpp::randint(len) / 1;

  for (auto i = 0; i < trials; i++) {
    auto idx = psz::testutils::cpp::randint(len);
    a[idx] = psz::testutils::cpp::randint(INT32_MAX);
  }

  // CPU counting nnz
  auto nnz_ref = 0;
  for (auto i = 0; i < len; i++) {
    if (a[i] != 0) nnz_ref += 1;
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  ////////////////////////////////////////////////////////////////

  // psz::spv_gather<PROPER_RUNTIME, T, uint32_t>(
  //     a, len, val, idx, &nnz, &ms, stream);

  psz::spv_gather_naive<PROPER_RUNTIME>(
      a, len, 0, val, idx, d_nnz, psz::criterion::gpu::eq<T>(), &ms, stream);
  nnz = *d_nnz;

  cudaStreamSynchronize(stream);

  if (nnz != nnz_ref) {
    std::cout << "nnz_ref: " << nnz_ref << std::endl;
    std::cout << "nnz: " << nnz << std::endl;
    std::cerr << "nnz != nnz_ref" << std::endl;
    return -1;
  }

  // psz::spv_scatter<PROPER_RUNTIME, T, uint32_t>(
  //     val, idx, nnz, da, &ms, stream);
  psz::spv_scatter_naive<PROPER_RUNTIME, T, uint32_t>(val, idx, nnz, da, &ms, stream);

  cudaStreamSynchronize(stream);

  ////////////////////////////////////////////////////////////////

  bool same = true;

  for (auto i = 0; i < len; i++) {
    if (a[i] != da[i]) {
      same = false;
      break;
    }
  }

  cudaFree(a);
  cudaFree(da);
  cudaFree(val);
  cudaFree(idx);

  cudaStreamDestroy(stream);

  if (same) {
    std::cout << "[psz::test::info] decode correct" << std::endl;
    return 0;
  }
  else {
    std::cout << "[psz::test::ERR] DECODE NOT CORRECT" << std::endl;
    return -1;
  }
}
