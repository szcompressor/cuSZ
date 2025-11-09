#ifndef PSZ_KERNEL_IMPL_SPVN_DPCPP_INL
#define PSZ_KERNEL_IMPL_SPVN_DPCPP_INL

#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "cusz/type.h"
#include "kernel/spvn.hh"

namespace psz {

template <typename T, typename M = u4>
void KERNEL_SYCL_spvn_scatter(
    T* val, M* idx, int const nnz, T* out, const sycl::nd_item<3>& item_ct1)
{
  auto tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

  if (tid < nnz) {
    int dst_idx = idx[tid];
    out[dst_idx] = val[tid];
  }
}

template <typename T, typename M = u4, typename ValIdx = _portable::compact_cell<T, M> >
void KERNEL_SYCL_spvn_scatter_v2(
    ValIdx* val_idx, int const nnz, T* out, const sycl::nd_item<3>& item_ct1)
{
  auto tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

  if (tid < nnz) {
    auto [val, idx] = val_idx[tid];
    out[idx] = val;
  }
}

}  // namespace psz

template <typename T, typename M>
int psz::module::GPU_scatter<T, M>::kernel(
    T* val, M* idx, int const nnz, T* out, f4* milliseconds, void* stream)
{
  auto grid_dim = (nnz - 1) / 128 + 1;
  auto q = (sycl::queue*)stream;

  sycl::event e = q->submit([&](sycl::handler& cgh) {
    auto val_ct0 = val;
    auto idx_ct1 = idx;
    auto nnz_ct2 = nnz;
    auto out_ct3 = out;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, grid_dim) * sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
          psz::KERNEL_SYCL_spvn_scatter<T, M>(val_ct0, idx_ct1, nnz_ct2, out_ct3, item_ct1);
        });
  });
  e.wait();
  SYCL_TIME_DELTA(e, *milliseconds);
}

template <typename T, typename M>
int psz::module::GPU_scatter<T, M>::kernel_v2(
    typename GPU_scatter<T, M>::ValIdx* val_idx, int const nnz, T* out, void* stream)
{
  auto grid_dim = (nnz - 1) / 128 + 1;
  auto q = (sycl::queue*)stream;

  sycl::event e = q->submit([&](sycl::handler& cgh) {
    auto val_idx_ct0 = val_idx;
    auto nnz_ct2 = nnz;
    auto out_ct3 = out;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, grid_dim) * sycl::range<3>(1, 1, 128), sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
          psz::KERNEL_SYCL_spvn_scatter_v2<T, M>(val_idx_ct0, nnz_ct2, out_ct3, item_ct1);
        });
  });
  e.wait();
  SYCL_TIME_DELTA(e, *milliseconds);
}

#endif /* PSZ_KERNEL_IMPL_SPVN_DPCPP_INL */
