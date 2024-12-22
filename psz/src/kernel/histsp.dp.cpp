#include "detail/histsp.dp.inl"

#include <chrono>
#include <cstdint>
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "module/cxx_module.hh"

namespace psz::module {

template <typename E>
int GPU_histogram_Cauchy(
    E* in, size_t const inlen, uint32_t* out_hist, uint16_t const outlen,
    float* milliseconds, void* queue)
{
  auto chunk = 32768;
  auto num_chunks = (inlen - 1) / chunk + 1;
  auto num_workers = 256;  // n SIMD-32
  using FREQ = uint32_t;

  /*
  DPCT1049:25: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  sycl::event e = (dpct::queue_ptr)queue->submit([&](sycl::handler& cgh) {
    /*
    DPCT1083:103: The size of local memory in the migrated code may be
    different from the original code. Check that the allocated memory size in
    the migrated code is correct.
    */
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
        sycl::range<1>(sizeof(FREQ) * outlen), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, num_chunks) *
                sycl::range<3>(1, 1, num_workers),
            sycl::range<3>(1, 1, num_workers)),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
          histsp_multiwarp<E>(
              in, inlen, chunk, out_hist, outlen, item_ct1,
              dpct_local_acc_ct1.get_pointer(), outlen / 2);
        });
  });

  e.wait();
  SYCL_TIME_DELTA(e, *milliseconds);

  return 0;
}

}  // namespace psz::module

#define INIT_HISTSP_1API(E)                                                  \
  template int psz::module::GPU_histogram_Cauchy<E>(                         \
      E * in, size_t const inlen, uint32_t* out_hist, uint16_t const outlen, \
      float* milliseconds, void* queue);

// INIT_HISTSP_1API(float)
INIT_HISTSP_1API(uint8_t)
INIT_HISTSP_1API(uint16_t)
INIT_HISTSP_1API(uint32_t)

#undef INIT_HISTSP_1API
