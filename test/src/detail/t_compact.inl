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
#include "busyheader.hh"
#include "mem/compact.hh"
#include "port.hh"
#include "cusz/type.h"

template <
    typename T, int TileDim = 256, typename CompactVal = T,
    typename CompactIdx = uint32_t, typename CompactNum = uint32_t>
__global__ void test_compaction1(
    T* in, uint32_t len, CompactVal* cval, CompactIdx* cidx, CompactNum* cn)
{
  auto id = blockIdx.x * TileDim + threadIdx.x;

  if (id < len) {
    auto delta = in[id] - (id > 0 ? in[id - 1] : 0) / 1e-3;
    auto predicate = [&]() { return abs(delta) > 512; };

    if (predicate()) {
      auto cur_idx = atomicAdd(cn, 1);
      cval[cur_idx] = delta;
      cidx[cur_idx] = id;
    }
  }
  // end of kernel
}

template <
    typename T, int TileDim = 256,
    typename Compact = typename CompactDram<PROPER_GPU_BACKEND, T>::Compact>
__global__ void test_compaction2(T* in, uint32_t len, Compact compact)
{
  auto id = blockIdx.x * TileDim + threadIdx.x;

  if (id < len) {
    auto delta = in[id] - (id > 0 ? in[id - 1] : 0) / 1e-3;
    auto predicate = [&]() { return abs(delta) > 512; };

    if (predicate()) {
      auto cur_idx = atomicAdd(compact.d_num, 1);
      compact.d_val[cur_idx] = delta;
      compact.d_val[cur_idx] = id;
    }
  }
  // end of kernel
}

template <typename T, typename Compact = CompactSerial<T>>
void test_compaction_serial(T* in, uint32_t len, Compact out)
{
  for (auto id = 0; id < len; id++) {
    auto delta = in[id] - (id > 0 ? in[id - 1] : 0) / 1e-3;
    auto predicate = [&]() { return abs(delta) > 512; };

    if (predicate()) {
      auto cur_idx = out.num()++;
      out.idx(cur_idx) = id;
      out.val(cur_idx) = delta;
    }
  }
}

bool f()
{

using CompactGpu = typename CompactDram<PROPER_GPU_BACKEND, float>::Compact;
using CompactSeq = typename CompactDram<SEQ, float>::Compact;

  constexpr auto TilDim = 256;

  // auto len       = (1u << 20) - 56;
  auto len = 256;
  auto block_dim = TilDim;
  auto grid_dim = (len - 1) / TilDim + 1;

  float* in;
  GpuMallocManaged(&in, sizeof(float) * len);
  psz::testutils::cu_hip::rand_array(in, len);

  CompactGpu out_test1;
  out_test1.reserve_space(len / 2).malloc().mallochost();

  CompactGpu out_test2;
  out_test2.reserve_space(len / 2).malloc().mallochost();

  CompactSeq out_ref;
  out_ref.reserve_space(len / 2).malloc();

  test_compaction1<float, TilDim><<<grid_dim, block_dim>>>(
      in, len, out_test1.d_val, out_test1.d_idx, out_test1.d_num);
  GpuDeviceSync();

  test_compaction2<float, TilDim><<<grid_dim, block_dim>>>(in, len, out_test2);
  GpuDeviceSync();

  cout << endl;

  out_test1.make_host_accessible();
  out_test2.make_host_accessible();

  test_compaction_serial<float>(in, len, out_ref);

  cout << "CPU #outlier:\t" << out_ref.num_outliers() << endl;
  cout << "GPU (plain)  #outlier:\t" << out_test1.num_outliers() << endl;
  cout << "GPU (struct) #outlier:\t" << out_test2.num_outliers() << endl;

  GpuFree(in);
  out_test1.free().freehost();
  out_test2.free().freehost();
  out_ref.free();

  return (out_ref.num_outliers() == out_test1.num_outliers()) and
         (out_ref.num_outliers() == out_test2.num_outliers());
}