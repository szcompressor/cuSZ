/**
 * @file compact.cu.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2022-12-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef F712F74C_7488_4445_83EE_EE7F88A64BBA
#define F712F74C_7488_4445_83EE_EE7F88A64BBA

#include <cuda_runtime.h>

#include <stdexcept>

#include "mem/memseg_cxx.hh"

namespace psz {
namespace detail {
namespace cuda {

template <typename T>
struct CompactGpuDram {
 private:
  static const cudaMemcpyKind h2d = cudaMemcpyHostToDevice;
  static const cudaMemcpyKind d2h = cudaMemcpyDeviceToHost;

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
    cudaMalloc(&d_val, sizeof(T) * reserved_len);
    cudaMalloc(&d_idx, sizeof(uint32_t) * reserved_len);
    cudaMalloc(&d_num, sizeof(uint32_t) * 1);
    cudaMemset(d_num, 0x0, sizeof(T) * 1);  // init d_val

    return *this;
  }

  CompactGpuDram& mallochost()
  {
    cudaMallocHost(&h_val, sizeof(T) * reserved_len);
    cudaMallocHost(&h_idx, sizeof(uint32_t) * reserved_len);

    return *this;
  }

  CompactGpuDram& free()
  {
    cudaFree(d_idx), cudaFree(d_val), cudaFree(d_num);
    return *this;
  }

  CompactGpuDram& freehost()
  {
    cudaFreeHost(h_idx), cudaFreeHost(h_val);
    return *this;
  }

  // memcpy
  CompactGpuDram& make_host_accessible(cudaStream_t stream = 0)
  {
    cudaMemcpyAsync(&h_num, d_num, 1 * sizeof(uint32_t), d2h, stream);
    cudaStreamSynchronize(stream);
    // cudaMemcpyAsync(h_val, d_val, sizeof(T) * (h_num), d2h, stream);
    // cudaMemcpyAsync(h_idx, d_idx, sizeof(uint32_t) * (h_num), d2h, stream);
    // cudaStreamSynchronize(stream);

    if (h_num > reserved_len)
      throw std::runtime_error(
          "[psz::err::compact] Too many outliers exceed the maximum allocated "
          "buffer.");

    return *this;
  }

  CompactGpuDram& control(
      std::vector<pszmem_control> control_stream,
      cudaStream_t stream = nullptr)
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

}  // namespace cuda
}  // namespace detail
}  // namespace psz

#endif /* F712F74C_7488_4445_83EE_EE7F88A64BBA */
