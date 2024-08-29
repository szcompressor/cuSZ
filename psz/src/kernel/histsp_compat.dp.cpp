#include <chrono>
#include <cstdint>
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "detail/histsp.dp.inl"
#include "module/cxx_module.hh"
#include "utils/timer.hh"

namespace psz::dpcpp {

template <typename T, typename FQ>
int GPU_histogram_sparse(
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

}  // namespace psz::dpcpp

#define SPECIALIZE_PSZCXX_COMPAT_MODULE_HIST_CAUCHY(E)                       \
  template <>                                                                \
  int pszcxx_compat_histogram_cauchy<psz_policy::ONEAPI, E, uint32_t>(       \
      E * in, uint32_t inlen, uint32_t * out_hist, uint32_t outlen,          \
      float* milliseconds, void* stream)                                     \
  {                                                                          \
    return psz::dpcpp::GPU_histogram_sparse<E, uint32_t>(                    \
        in, inlen, out_hist, outlen, milliseconds, (dpct::queue_ptr)stream); \
  }

// SPECIALIZE_PSZCXX_COMPAT_MODULE_HIST_CAUCHY(float)
// SPECIALIZE_PSZCXX_COMPAT_MODULE_HIST_CAUCHY(uint8_t)
// SPECIALIZE_PSZCXX_COMPAT_MODULE_HIST_CAUCHY(uint16_t)
SPECIALIZE_PSZCXX_COMPAT_MODULE_HIST_CAUCHY(uint32_t)

#undef SPECIALIZE_PSZCXX_COMPAT_MODULE_HIST_CAUCHY
