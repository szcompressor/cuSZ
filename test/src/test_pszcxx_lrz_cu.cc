#include <cuda_runtime.h>

#include "cusz/type.h"
#include "experimental/core_mem.hh"
#include "pszcxx.hh"

template <typename T = f4>
bool test()
{
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto linear = 3600 * 1800;

  auto data = malloc_device<T>(linear);
  auto xdata = malloc_device<T>(linear);
  auto eq = malloc_device<u4>(linear);
  size_t outlier_reserved = linear / 5;
  auto outlier_val = malloc_device<T>(outlier_reserved);
  auto outlier_idx = malloc_device<u4>(outlier_reserved);
  auto outlier_num = malloc_device<u4>(1);

  pszlen shape1d = {6480000, 1, 1};
  pszlen shape2d = {3600, 1800, 1};
  pszlen shape3d = {360, 180, 100};

  _2401::pszcxx_predict_lorenzo<T>(
      {data, shape1d}, {.eb = 1e-3, .radius = 512}, {eq, shape1d},
      {outlier_val, outlier_idx, outlier_num, outlier_reserved}, stream);

  _2401::pszcxx_reverse_predict_lorenzo<T>(
      {eq, shape1d}, {xdata, shape1d}, {.eb = 1e-3, .radius = 512},
      {xdata, shape1d}, stream);

  free_device(data);
  free_device(xdata);
  free_device(eq);

  cudaStreamDestroy(stream);

  return true;
}

int main()
{
  test<f4>();

  return 0;
}