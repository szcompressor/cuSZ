#ifndef BDA1F851_E0AB_4877_9FB2_20BEAC0328F2
#define BDA1F851_E0AB_4877_9FB2_20BEAC0328F2

#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

namespace psz {

namespace dpcpp {

template <typename T = float, typename FP = T, int BLOCK = 256, int SEQ = 4>
void dryrun_kernel(
    T* in, T* out, size_t len, FP ebx2_r, FP ebx2,
    const sycl::nd_item<3>& item_ct1, T* shmem)
{
  {
    constexpr auto NTHREAD = BLOCK / SEQ;

    auto id_base = item_ct1.get_group(2) * BLOCK;

#pragma unroll
    for (auto i = 0; i < SEQ; i++) {
      auto id = id_base + item_ct1.get_local_id(2) + i * NTHREAD;
      if (id < len) {
        shmem[item_ct1.get_local_id(2) + i * NTHREAD] =
            sycl::round(in[id] * ebx2_r) * ebx2;
        out[id] = shmem[item_ct1.get_local_id(2) + i * NTHREAD];
      }
    }
  }
}

template <typename T>
void dryrun(size_t len, T* original, T* reconst, PROPER_EB eb, void* stream)
{
  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  auto ebx2_r = 1 / (eb * 2);
  auto ebx2 = eb * 2;

  auto queue = (sycl::queue*)stream;

  /*
  DPCT1049:75: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(queue->get_device(), {sycl::aspect::fp64});
    queue->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<T, 1> shmem_acc_ct1(sycl::range<1>(256), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(
              div(len, 256) * sycl::range<3>(1, 1, 256),
              sycl::range<3>(1, 1, 256)),
          [=](sycl::nd_item<3> item_ct1) {
            dryrun_kernel(
                original, reconst, len, ebx2_r, ebx2, item_ct1,
                (T*)shmem_acc_ct1.get_pointer());
          });
    });
  }

  queue->wait();
}

}  // namespace dpcpp

}  // namespace psz

#endif /* BDA1F851_E0AB_4877_9FB2_20BEAC0328F2 */
