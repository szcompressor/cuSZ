#include "kernel/spv.hh"

#include <chrono>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

#include "utils/timer.hh"

namespace psz {

template <typename T, typename M>
void spv_gather_dpl(
    T* in, szt const in_len, T* d_val, M* d_idx, int* nnz, f4* milliseconds,
    dpct::queue_ptr q)
{
  oneapi::dpl::counting_iterator<uint32_t> zero(0);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(q);

  // find out the indices
  *nnz =
      dpct::copy_if(
          oneapi::dpl::execution::make_device_policy(*q), zero, zero + in_len,
          in, d_idx, [](auto t) { return std::get<0>(t) != 0; }) -
      d_idx;

  // fetch corresponding values
  std::copy(
      oneapi::dpl::execution::make_device_policy(*q),
      oneapi::dpl::make_permutation_iterator(in, d_idx),
      oneapi::dpl::make_permutation_iterator(in + *nnz, d_idx + *nnz), d_val);

  STOP_GPUEVENT_RECORDING(q);
  TIME_ELAPSED_GPUEVENT(milliseconds);
  DESTROY_GPUEVENT_PAIR;
}

template <typename T, typename M>
void spv_scatter_dpl(
    T* d_val, M* d_idx, int const nnz, T* decoded, f4* milliseconds,
    dpct::queue_ptr q)
{
  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(q);

  dpct::scatter(
      oneapi::dpl::execution::make_device_policy(*q), d_val, d_val + nnz,
      d_idx, decoded);

  STOP_GPUEVENT_RECORDING(q);
  TIME_ELAPSED_GPUEVENT(milliseconds);
  DESTROY_GPUEVENT_PAIR;
}

}  // namespace psz

#define SPECIALIZE_SPV(T, M)                                              \
  template <>                                                             \
  void psz::spv_gather<ONEAPI, T, M>(                                     \
      T * in, szt const in_len, T* d_val, M* d_idx, int* nnz,             \
      f4* milliseconds, void* q)                                          \
  {                                                                       \
    psz::spv_gather<ONEAPI, T, M>(                                        \
        in, in_len, d_val, d_idx, nnz, milliseconds, (dpct::queue_ptr)q); \
  }                                                                       \
  template <>                                                             \
  void psz::spv_scatter<ONEAPI, T, M>(                                    \
      T * d_val, M * d_idx, int const nnz, T* decoded, f4* milliseconds,  \
      void* q)                                                            \
  {                                                                       \
    psz::spv_scatter_dpl(                                                 \
        d_val, d_idx, nnz, decoded, milliseconds, (dpct::queue_ptr)q);    \
  }

SPECIALIZE_SPV(f4, u4)
// SPECIALIZE_SPV(f8, u4)

#undef SPECIALIZE_SPV
