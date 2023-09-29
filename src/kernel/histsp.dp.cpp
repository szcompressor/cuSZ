#include "detail/histsp.dp.inl"

#include <chrono>
#include <cstdint>
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "kernel/histsp.hh"
#include "utils/timer.hh"

namespace psz {
namespace detail {

template <typename T, typename FQ>
int histsp_dpcpp(
    T* in, uint32_t inlen, FQ* out_hist, uint32_t outlen, float* milliseconds,
    dpct::queue_ptr queue)
{
  auto chunk = 32768;
  auto num_chunks = (inlen - 1) / chunk + 1;
  auto num_workers = 256;  // n SIMD-32

  // CREATE_GPUEVENT_PAIR;
  // START_GPUEVENT_RECORDING(queue);

  /*
  DPCT1049:25: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  sycl::event e = queue->submit([&](sycl::handler& cgh) {
    /*
    DPCT1083:103: The size of local memory in the migrated code may be
    different from the original code. Check that the allocated memory size in
    the migrated code is correct.
    */
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
        sycl::range<1>(sizeof(FQ) * outlen), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, num_chunks) *
                sycl::range<3>(1, 1, num_workers),
            sycl::range<3>(1, 1, num_workers)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
          histsp_multiwarp<T, FQ>(
              in, inlen, chunk, out_hist, outlen, item_ct1,
              dpct_local_acc_ct1.get_pointer(), outlen / 2);
        });
  });

  e.wait();
  SYCL_TIME_DELTA(e, *milliseconds);

  // STOP_GPUEVENT_RECORDING(queue);
  // TIME_ELAPSED_GPUEVENT(milliseconds);
  // DESTROY_GPUEVENT_PAIR;

  return 0;
}

}  // namespace detail
}  // namespace psz

#define SPECIALIZE_DPCPP(E)                                                  \
  template <>                                                                \
  int psz::histsp<pszpolicy::ONEAPI, E, uint32_t>(                           \
      E * in, uint32_t inlen, uint32_t * out_hist, uint32_t outlen,          \
      float* milliseconds, void* stream)                                     \
  {                                                                          \
    return psz::detail::histsp_dpcpp<E, uint32_t>(                           \
        in, inlen, out_hist, outlen, milliseconds, (dpct::queue_ptr)stream); \
  }

// SPECIALIZE_DPCPP(float)
// SPECIALIZE_DPCPP(uint8_t)
// SPECIALIZE_DPCPP(uint16_t)
SPECIALIZE_DPCPP(uint32_t)

#undef SPECIALIZE_CUDA
