/**
 * @file compact_cuda.inl
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
#include <stdint.h>
#include <stdlib.h>

#include <cstring>

#include "compaction.hh"
#include "mem/memseg_cxx.hh"

// TODO filename -> `compaction`
template <typename T>
struct CompactCudaDram {
 private:
  static const cudaMemcpyKind h2d = cudaMemcpyHostToDevice;
  static const cudaMemcpyKind d2h = cudaMemcpyDeviceToHost;

 public:
  using type = T;

  // `h_` for host-accessible
  T *val, *h_val;
  uint32_t *idx, *h_idx;
  uint32_t *num, h_num{0};
  size_t reserved_len;

  // CompactCudaDram() {}
  // ~CompactCudaDram() {}

  CompactCudaDram& reserve_space(size_t _reserved_len)
  {
    reserved_len = _reserved_len;
    return *this;
  }

  CompactCudaDram& malloc()
  {
    cudaMalloc(&val, sizeof(T) * reserved_len);
    cudaMalloc(&idx, sizeof(uint32_t) * reserved_len);
    cudaMalloc(&num, sizeof(uint32_t) * 1);
    cudaMemset(num, 0x0, sizeof(T) * 1);  // init val

    return *this;
  }

  CompactCudaDram& mallochost()
  {
    cudaMallocHost(&h_val, sizeof(T) * reserved_len);
    cudaMallocHost(&h_idx, sizeof(uint32_t) * reserved_len);

    return *this;
  }

  CompactCudaDram& free()
  {
    cudaFree(idx), cudaFree(val), cudaFree(num);
    return *this;
  }

  CompactCudaDram& freehost()
  {
    cudaFreeHost(h_idx), cudaFreeHost(h_val);
    return *this;
  }

  // memcpy
  CompactCudaDram& make_host_accessible(cudaStream_t stream = 0)
  {
    cudaMemcpyAsync(&h_num, num, 1 * sizeof(uint32_t), d2h, stream);
    cudaStreamSynchronize(stream);
    cudaMemcpyAsync(h_val, val, sizeof(T) * (h_num), d2h, stream);
    cudaMemcpyAsync(h_idx, idx, sizeof(uint32_t) * (h_num), d2h, stream);
    cudaStreamSynchronize(stream);

    return *this;
  }

  CompactCudaDram& control(
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
};

#endif /* F712F74C_7488_4445_83EE_EE7F88A64BBA */
