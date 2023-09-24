/**
 * @file compact.hip.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2022-12-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef E1192862_6E24_41A9_87D6_6B0BC7699283
#define E1192862_6E24_41A9_87D6_6B0BC7699283

#include <hip/hip_runtime.h>

#include "mem/memseg_cxx.hh"

namespace psz::detail::hip {

template <typename T>
struct CompactGpuDram {
 private:
  static const hipMemcpyKind h2d = hipMemcpyHostToDevice;
  static const hipMemcpyKind d2h = hipMemcpyDeviceToHost;

 public:
  using type = T;

  // `h_` for host-accessible
  T *d_val, *h_val;
  uint32_t *d_idx, *h_idx;
  uint32_t *d_num, h_num{0};
  size_t reserved_len;

  // CompactGpuDram() {}
  // ~CompactGpuDram() {}

  CompactGpuDram& reserve_space(size_t _reserved_len)
  {
    reserved_len = _reserved_len;
    return *this;
  }

  CompactGpuDram& malloc()
  {
    hipMalloc(&d_val, sizeof(T) * reserved_len);
    hipMalloc(&d_idx, sizeof(uint32_t) * reserved_len);
    hipMalloc(&d_num, sizeof(uint32_t) * 1);
    hipMemset(d_num, 0x0, sizeof(T) * 1);  // init d_val

    return *this;
  }

  CompactGpuDram& mallochost()
  {
    hipHostMalloc(&h_val, sizeof(T) * reserved_len);
    hipHostMalloc(&h_idx, sizeof(uint32_t) * reserved_len);

    return *this;
  }

  CompactGpuDram& free()
  {
    hipFree(d_idx), hipFree(d_val), hipFree(d_num);
    return *this;
  }

  CompactGpuDram& freehost()
  {
    hipHostFree(h_idx), hipHostFree(h_val);
    return *this;
  }

  // memcpy
  CompactGpuDram& make_host_accessible(hipStream_t stream = 0)
  {
    hipMemcpyAsync(&h_num, d_num, 1 * sizeof(uint32_t), d2h, stream);
    hipStreamSynchronize(stream);
    hipMemcpyAsync(h_val, d_val, sizeof(T) * (h_num), d2h, stream);
    hipMemcpyAsync(h_idx, d_idx, sizeof(uint32_t) * (h_num), d2h, stream);
    hipStreamSynchronize(stream);

    return *this;
  }

  CompactGpuDram& control(
      std::vector<pszmem_control> control_stream, hipStream_t stream = nullptr)
  {
    for (auto& c : control_stream) {
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
  T* val() { return d_val; }
  uint32_t* idx() { return d_idx; }
  uint32_t* num() { return d_num; }
};

}  // namespace psz::detail::hip

#endif /* E1192862_6E24_41A9_87D6_6B0BC7699283 */
