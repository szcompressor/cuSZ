#ifndef C4963A4F_B1D2_4A1E_8ED2_5401B7652A98
#define C4963A4F_B1D2_4A1E_8ED2_5401B7652A98

#include <dpct/dpct.hpp>
#include <stdexcept>
#include <sycl/sycl.hpp>

#include "mem/memseg_cxx.hh"

namespace psz::detail::dpcpp {

template <typename T>
struct CompactGpuDram {
 private:
  static const dpct::memcpy_direction h2d = dpct::host_to_device;
  static const dpct::memcpy_direction d2h = dpct::device_to_host;

 public:
  using type = T;

  // `h_` for host-accessible
  T *d_val, *h_val;
  uint32_t *d_idx, *h_idx;
  uint32_t *d_num, h_num{0};
  size_t reserved_len;

  // CompactGpuDram() {}
  // ~CompactGpuDram() {}

  CompactGpuDram &reserve_space(size_t _reserved_len)
  {
    reserved_len = _reserved_len;
    return *this;
  }

  CompactGpuDram &malloc()
  {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    d_val = (T *)sycl::malloc_device(sizeof(T) * reserved_len, q_ct1);
    d_idx = sycl::malloc_device<uint32_t>(reserved_len, q_ct1);
    d_num = sycl::malloc_device<uint32_t>(1, q_ct1);
    q_ct1.memset(d_num, 0x0, sizeof(T) * 1).wait();  // init d_val

    return *this;
  }

  CompactGpuDram &mallochost()
  {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    h_val = (T *)sycl::malloc_host(sizeof(T) * reserved_len, q_ct1);
    h_idx = sycl::malloc_host<uint32_t>(reserved_len, q_ct1);

    return *this;
  }

  CompactGpuDram &free()
  {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    sycl::free(d_idx, q_ct1), sycl::free(d_val, q_ct1),
        sycl::free(d_num, q_ct1);
    return *this;
  }

  CompactGpuDram &freehost()
  {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    sycl::free(h_idx, q_ct1), sycl::free(h_val, q_ct1);
    return *this;
  }

  // memcpy
  CompactGpuDram &make_host_accessible(
      dpct::queue_ptr stream = &dpct::get_default_queue())
  {
    stream->memcpy(&h_num, d_num, 1 * sizeof(uint32_t));
    stream->wait();
    // cudaMemcpyAsync(h_val, d_val, sizeof(T) * (h_num), d2h, stream);
    // cudaMemcpyAsync(h_idx, d_idx, sizeof(uint32_t) * (h_num), d2h, stream);
    // cudaStreamSynchronize(stream);

    if (h_num > reserved_len)
      throw std::runtime_error(
          "[psz::err::compact] Too many outliers exceed the maximum allocated "
          "buffer.");

    return *this;
  }

  CompactGpuDram &control(
      std::vector<pszmem_control> control_stream,
      dpct::queue_ptr stream = &dpct::get_default_queue())
  {
    for (auto &c : control_stream) {
      if (c == Malloc)
        malloc();
      else if (c == MallocHost)
        mallochost();
      else if (c == Free)
        free();
      else if (c == FreeHost)
        freehost();
      else if (c == D2H)
        make_host_accessible(stream);
    }

    return *this;
  }

  // accessor
  uint32_t num_outliers() { return h_num; }
  T *val() { return d_val; }
  uint32_t *idx() { return d_idx; }
  uint32_t *num() { return d_num; }
};

}  // namespace psz::detail::dpcpp

#endif /* C4963A4F_B1D2_4A1E_8ED2_5401B7652A98 */
