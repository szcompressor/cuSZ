#ifndef FC6B1F0D_E138_4DF5_8AC9_896900D1BE21
#define FC6B1F0D_E138_4DF5_8AC9_896900D1BE21

#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "cusz/type.h"

namespace psz {
namespace dpcpp {

template <typename T, typename Criterion, typename M = u4>
void spvn_gather(
    T* in, szt const in_len, int const radius, T* cval, M* cidx, int* cn,
    Criterion criteria, const sycl::nd_item<3>& item_ct1)
{
  auto tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);

  if (tid < in_len) {
    auto d = in[tid];
    auto quantizable = criteria(d, radius);

    if (not quantizable) {
      auto cur_idx =
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
              cn, 1);
      cidx[cur_idx] = tid;
      cval[cur_idx] = d;
    }
  }
}

template <typename T, typename M = u4>
void spvn_scatter(
    T* val, M* idx, int const nnz, T* out, const sycl::nd_item<3>& item_ct1)
{
  auto tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);

  if (tid < nnz) {
    int dst_idx = idx[tid];
    out[dst_idx] = val[tid];
  }
}

}  // namespace dpcpp
}  // namespace psz

#endif /* FC6B1F0D_E138_4DF5_8AC9_896900D1BE21 */
