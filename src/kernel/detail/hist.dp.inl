#ifndef C6D3EFB0_47D3_40FB_A618_FA7412DC5376
#define C6D3EFB0_47D3_40FB_A618_FA7412DC5376

#include <chrono>
#include <cmath>
#include <cstdio>
#include <dpct/dpct.hpp>
#include <limits>
#include <sycl/sycl.hpp>

#include "typing.hh"
#include "utils/config.hh"
#include "utils/timer.hh"

#define MIN(a, b) ((a) < (b)) ? (a) : (b)
const static unsigned int WARP_SIZE = 32;

#define tix item_ct1.get_local_id(2)
#define tiy threadIdx.y
#define tiz threadIdx.z
#define bix blockIdx.x
#define biy blockIdx.y
#define biz blockIdx.z
#define bdx item_ct1.get_local_range(2)
#define bdy blockDim.y
#define bdz blockDim.z

namespace kernel {

template <typename Input>
void NaiveHistogram(
    Input in_data[], int out_freq[], int N, int symbols_per_thread,
    const sycl::nd_item<3> &item_ct1);

/* Copied from J. Gomez-Luna et al */
template <typename T, typename FREQ>
void p2013Histogram(
    T *, FREQ *, size_t, int, int, const sycl::nd_item<3> &item_ct1,
    uint8_t *dpct_local);

}  // namespace kernel

template <typename T>
void kernel::NaiveHistogram(
    T in_data[], int out_freq[], int N, int symbols_per_thread,
    const sycl::nd_item<3> &item_ct1)
{
  unsigned int i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                   item_ct1.get_local_id(2);
  unsigned int j;
  if (i * symbols_per_thread < N) {  // if there is a symbol to count,
    for (j = i * symbols_per_thread; j < (i + 1) * symbols_per_thread; j++) {
      if (j < N) {
        unsigned int item = in_data[j];  // Symbol to count
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &out_freq[item], 1);  // update bin count by 1
      }
    }
  }
}

template <typename T, typename FREQ>
void kernel::p2013Histogram(
    T *in_data, FREQ *out_freq, size_t N, int nbin, int R,
    const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local)
{
  // static_assert(
  //     std::numeric_limits<T>::is_integer and (not
  //     std::numeric_limits<T>::is_signed), "T must be `unsigned integer` type
  //     of {1,2,4} bytes");

  auto Hs = (int *)dpct_local;

  const unsigned int warp_id = (int)(tix / WARP_SIZE);
  const unsigned int lane = tix % WARP_SIZE;
  const unsigned int warps_block = bdx / WARP_SIZE;
  const unsigned int off_rep = (nbin + 1) * (tix % R);
  const unsigned int begin =
      (N / warps_block) * warp_id + WARP_SIZE * item_ct1.get_group(2) + lane;
  unsigned int end = (N / warps_block) * (warp_id + 1);
  const unsigned int step = WARP_SIZE * item_ct1.get_group_range(2);

  // final warp handles data outside of the warps_block partitions
  if (warp_id >= warps_block - 1) end = N;

  for (unsigned int pos = tix; pos < (nbin + 1) * R; pos += bdx) Hs[pos] = 0;
  /*
  DPCT1065:76: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  for (unsigned int i = begin; i < end; i += step) {
    int d = in_data[i];
    d = d <= 0 and d >= nbin ? nbin / 2 : d;
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &Hs[off_rep + d], 1);
  }
  /*
  DPCT1065:77: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  for (unsigned int pos = tix; pos < nbin; pos += bdx) {
    int sum = 0;
    for (int base = 0; base < (nbin + 1) * R; base += nbin + 1) {
      sum += Hs[base + pos];
    }
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        out_freq + pos, sum);
  }
}

namespace psz {
namespace cuda_hip_compat {

template <typename T>
psz_error_status hist_default(
    T *in, size_t const inlen, uint32_t *out_hist, int const outlen,
    float *milliseconds, dpct::queue_ptr queue)
try {
  int device_id, max_bytes, num_SMs;
  int items_per_thread, r_per_block, grid_dim, block_dim, shmem_use;

  device_id = dpct::dev_mgr::instance().current_device_id();
  num_SMs =
      dpct::dev_mgr::instance().get_device(device_id).get_max_compute_units();

  auto query_maxbytes = [&]() {
    int max_bytes_opt_in;
    GpuDeviceGetAttribute(
        &max_bytes, GpuDevAttrMaxSharedMemoryPerBlock, device_id);

    // account for opt-in extra shared memory on certain architectures
    GpuDeviceGetAttribute(
        &max_bytes_opt_in, GpuDevAttrMaxSharedMemoryPerBlockOptin, device_id);
    max_bytes = std::max(max_bytes, max_bytes_opt_in);

    // config kernel attribute
    /*
    DPCT1007:102: Migration of cudaFuncSetAttribute is not supported.
    */
    GpuFuncSetAttribute(
        (void *)kernel::p2013Histogram<T, uint32_t>,
        (GpuFuncAttribute)GpuFuncAttributeMaxDynamicSharedMemorySize,
        max_bytes);
  };

  auto optimize_launch = [&]() {
    items_per_thread = 1;
    r_per_block = (max_bytes / sizeof(int)) / (outlen + 1);
    grid_dim = num_SMs;
    // fits to size
    block_dim =
        ((((inlen / (grid_dim * items_per_thread)) + 1) / 64) + 1) * 64;
    while (block_dim > 1024) {
      if (r_per_block <= 1) { block_dim = 1024; }
      else {
        r_per_block /= 2;
        grid_dim *= 2;
        block_dim =
            ((((inlen / (grid_dim * items_per_thread)) + 1) / 64) + 1) * 64;
      }
    }
    /*
    DPCT1083:79: The size of local memory in the migrated code may be different
    from the original code. Check that the allocated memory size in the
    migrated code is correct.
    */
    shmem_use = ((outlen + 1) * r_per_block) * sizeof(int);
  };

  query_maxbytes();
  optimize_launch();

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(queue);

  /*
  DPCT1049:78: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  queue->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
        sycl::range<1>(shmem_use), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, grid_dim) * sycl::range<3>(1, 1, block_dim),
            sycl::range<3>(1, 1, block_dim)),
        [=](sycl::nd_item<3> item_ct1) {
          kernel::p2013Histogram(
              in, out_hist, inlen, outlen, r_per_block, item_ct1,
              dpct_local_acc_ct1.get_pointer());
        });
  });

  STOP_GPUEVENT_RECORDING(queue);
  TIME_ELAPSED_GPUEVENT(milliseconds);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace cuda_hip_compat
}  // namespace psz

#endif /* C6D3EFB0_47D3_40FB_A618_FA7412DC5376 */
