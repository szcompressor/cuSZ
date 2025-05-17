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

template <
    typename T, int TileDim = 256, typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t>
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

template <typename T, typename Compact = _portable::compact_seq<T>>
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
  using compact_gpu = _portable::compact_gpu<float>;
  using compact_seq = _portable::compact_seq<float>;

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
  out_ref.malloc();

  test_compaction1<float, TilDim><<<grid_dim, block_dim>>>(
      in, len, out_test1.d_val.get(), out_test1.d_idx.get(), out_test1.d_num.get());
  cudaDeviceSynchronize();

  // test_compaction2<float, TilDim><<<grid_dim, block_dim>>>(in, len, out_test2);
  // cudaDeviceSynchronize();

  cout << endl;

  test_compaction_serial<float>(in, len, out_ref);

  cout << "CPU #outlier:\t" << out_ref.num_outliers() << endl;
  cout << "GPU (plain)  #outlier:\t" << out_test1.num_outliers() << endl;
  cout << "GPU (struct) #outlier:\t" << out_test2.num_outliers() << endl;

  cudaFree(in);
  out_ref.free();

  return (out_ref.num_outliers() == out_test1.num_outliers()) and
         (out_ref.num_outliers() == out_test2.num_outliers());
}