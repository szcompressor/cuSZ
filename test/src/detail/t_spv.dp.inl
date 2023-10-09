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

#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "kernel/criteria.gpu.hh"
#include "kernel/spv.hh"

template <typename T = float>
int f()
{
  sycl::queue q(
      sycl::gpu_selector_v, sycl::property_list(
                                sycl::property::queue::in_order(),
                                sycl::property::queue::enable_profiling()));

  T* a;                // input
  T* da;               // decoded
  size_t len = 10000;  //
  T* val;              // intermeidate
  uint32_t* idx;       //
  int nnz;             //
  float ms;

  a = (T*)sycl::malloc_shared(sizeof(T) * len, q);
  da = (T*)sycl::malloc_shared(sizeof(T) * len, q);
  val = (T*)sycl::malloc_shared(sizeof(T) * len, q);
  idx = sycl::malloc_shared<uint32_t>(len, q);

  auto d_nnz = sycl::malloc_shared<int>(1, q);

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

  ////////////////////////////////////////////////////////////////

  // psz::spv_gather<PROPER_GPU_BACKEND, T, uint32_t>(
  //     a, len, val, idx, &nnz, &ms, &q);

  psz::spv_gather_naive<PROPER_GPU_BACKEND>(
      a, len, 0, val, idx, d_nnz, psz::criterion::gpu::eq<T>(), &ms, &q);
  nnz = *d_nnz;

  q.wait();

  if (nnz != nnz_ref) {
    std::cout << "nnz_ref: " << nnz_ref << std::endl;
    std::cout << "nnz: " << nnz << std::endl;
    std::cerr << "nnz != nnz_ref" << std::endl;
    return -1;
  }

  psz::spv_scatter<PROPER_GPU_BACKEND, T, uint32_t>(
      val, idx, nnz, da, &ms, &q);

  q.wait();

  ////////////////////////////////////////////////////////////////

  bool same = true;

  for (auto i = 0; i < len; i++) {
    if (a[i] != da[i]) {
      same = false;
      break;
    }
  }

  sycl::free(a, q);
  sycl::free(da, q);
  sycl::free(val, q);
  sycl::free(idx, q);

  if (same)
    return 0;
  else {
    std::cout << "decomp not okay" << std::endl;
    return -1;
  }
}
