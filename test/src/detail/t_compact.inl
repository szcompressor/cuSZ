/**
 * @file test_l1_compact.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-04-05
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include "../rand.hh"
#include "cusz/type.h"
#include "detail/busyheader.hh"
#include "mem/cxx_sp_cpu.h"
#include "mem/cxx_sp_gpu.h"

using compact_gpu = _portable::compact_GPU_DRAM2<float>;
using compact_seq = _portable::compact_CPU<float>;

template <
    typename T, int TileDim = 256, typename CompactValIdx = _portable::compact_GPU_DRAM2<T, u4>,
    typename CompactNum = uint32_t>
__global__ void test_compaction1(T* in, uint32_t len, CompactValIdx* cvalidx, CompactNum* cn)
{
  auto id = blockIdx.x * TileDim + threadIdx.x;

  if (id < len) {
    auto delta = in[id] - (id > 0 ? in[id - 1] : 0) / 1e-3;
    auto predicate = [&]() { return abs(delta) > 512; };

    if (predicate()) {
      auto cur_idx = atomicAdd(cn, 1);
      cvalidx[cur_idx] = {(float)delta, id};
    }
  }
  // end of kernel
}

template <typename T, typename Compact>
void test_compaction_serial(T* in, uint32_t len, Compact* out)
{
  for (auto id = 0u; id < len; id++) {
    auto delta = in[id] - (id > 0 ? in[id - 1] : 0) / 1e-3;
    auto predicate = [&]() { return abs(delta) > 512; };

    if (predicate()) {
      auto cur_idx = out->num()++;
      out->val_idx(cur_idx) = {(float)delta, id};
    }
  }
}

bool f()
{
  constexpr auto TilDim = 256;

  // auto len       = (1u << 20) - 56;
  auto len = 256;
  auto block_dim = TilDim;
  auto grid_dim = (len - 1) / TilDim + 1;

  float* in;
  cudaMallocManaged(&in, sizeof(float) * len);
  psz::testutils::cu_hip::rand_array(in, len);

  compact_gpu out_test1(len / 2);
  compact_gpu out_test2(len / 2);

  compact_seq out_ref(len / 2);

  test_compaction1<float, TilDim>
      <<<grid_dim, block_dim>>>(in, len, out_test1.val_idx_d(), out_test1.num_d());
  cudaDeviceSynchronize();

  // test_compaction2<float, TilDim><<<grid_dim, block_dim>>>(in, len, out_test2);
  // cudaDeviceSynchronize();

  cout << endl;

  test_compaction_serial<float>(in, len, &out_ref);

  cout << "CPU #outlier:\t" << out_ref.num() << endl;
  cout << "GPU (plain)  #outlier:\t" << out_test1.host_get_num() << endl;
  cout << "GPU (struct) #outlier:\t" << out_test2.host_get_num() << endl;

  cudaFree(in);

  return (out_ref.num() == out_test1.host_get_num()) and
         (out_ref.num() == out_test2.host_get_num());
}