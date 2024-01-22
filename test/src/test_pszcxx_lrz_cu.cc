#include <cuda_runtime.h>

#include "cusz/cxx_array.hh"
#include "cusz/type.h"
#include "experimental/core_mem.hh"
#include "module/cxx_module.hh"
#include "pszcxx.hh"
#include "utils/io.hh"
#include "utils/viewer/viewer.cu_hip.hh"

template <typename T = f4>
bool test(char* fname)
{
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  size_t linear = 3600 * 1800;
  size_t outlier_reserved = linear / 5;

  float* milliseconds;

  T *data, *xdata, *host_data, *host_xdata;
  u4* eq;
  T* outlier_val;
  u4 *outlier_idx, *outlier_num, *outlier_host_num;
  cudaMalloc(&data, sizeof(T) * linear);
  cudaMalloc(&xdata, sizeof(T) * linear);
  cudaMallocHost(&host_data, sizeof(T) * linear);
  cudaMallocHost(&host_xdata, sizeof(T) * linear);
  cudaMalloc(&eq, sizeof(u4) * linear);
  cudaMalloc(&outlier_val, sizeof(T) * outlier_reserved);
  cudaMalloc(&outlier_idx, sizeof(u4) * outlier_reserved);
  cudaMalloc(&outlier_num, sizeof(u4));
  cudaMallocHost(&outlier_host_num, sizeof(u4));

  // auto data = malloc_device<T>(linear);
  // auto xdata = malloc_device<T>(linear);
  // auto host_data = malloc_device<T>(linear);
  // auto host_xdata = malloc_device<T>(linear);
  // auto eq = malloc_device<u4>(linear);
  // auto outlier_val = malloc_device<T>(outlier_reserved);
  // auto outlier_idx = malloc_device<u4>(outlier_reserved);
  // auto outlier_num = malloc_device<u4>(1);
  // auto outlier_host_num = malloc_host<u4>(1);

  io::read_binary_to_array(fname, host_data, linear);

  cudaMemcpy(data, host_data, sizeof(T) * linear, cudaMemcpyHostToDevice);

  // for (auto i = 0; i < 20; i++) { cout << host_data[200 + i] << '\n'; }

  pszlen shape1d = {6480000, 1, 1};
  pszlen shape2d = {3600, 1800, 1};
  pszlen shape3d = {360, 180, 100};

  auto the_eb = 1e-2;
  auto the_data_size = shape2d;

  pszcompact_cxx<T> outlier_obj = {
      outlier_val, outlier_idx, outlier_num, outlier_host_num,
      outlier_reserved};

  _2401::pszcxx_predict_lorenzo<T>(
      {data, the_data_size}, {.eb = the_eb, .radius = 512}, {eq, shape1d},
      outlier_obj, stream);

  _2401::pszcxx_reverse_predict_lorenzo<T>(
      {eq, shape1d}, {xdata, shape1d}, {.eb = the_eb, .radius = 512},
      {xdata, the_data_size}, stream);

  // updated coding style
  // _2401::pszcxx_evaluate_quality_gpu<T>(
  //     {xdata, {linear, 1, 1}}, {data, {linear, 1, 1}});

  // fallback
  pszcxx_evaluate_quality_gpu<T>(xdata, data, linear);

  free_device(data);
  free_device(xdata);
  free_device(eq);

  cudaStreamDestroy(stream);

  return true;
}

int main(int argc, char** argv)
{
  if (argc < 2)
    printf(
        "Need to specify the absolute path to a CESM dataset, whose size "
        "(x,y)=(3600,1800).");
  else
    test<f4>(argv[1]);

  return 0;
}