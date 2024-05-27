#include <cuda_runtime.h>

#include "cusz/type.h"
#include "experimental/mem_multibackend.hh"
#include "mem/array_cxx.h"
#include "module/cxx_module.hh"
#include "pszcxx.hh"
#include "utils/io.hh"
#include "utils/viewer/viewer.cu_hip.hh"

using namespace portable;

template <typename T = f4, typename E = u2>
bool test(char* fname)
{
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  size_t linear = 3600 * 1800;
  size_t outlier_reserved = linear / 5;

  float* milliseconds;

  T *data, *xdata, *host_data, *host_xdata;
  E* eq;
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

  psz_len3 shape1d = {6480000, 1, 1};
  psz_len3 shape2d = {3600, 1800, 1};
  psz_len3 shape3d = {360, 180, 100};

  auto the_eb = 1e-2;
  auto the_data_size = shape2d;

  compact_array1<T> outlier_obj = {
      outlier_val, outlier_idx, outlier_num, outlier_host_num,
      outlier_reserved};

  float t_pred{0.0}, t_revpred{0};

  _2401::pszpred_lrz<T, E>::pszcxx_predict_lorenzo(
      {data, the_data_size}, {.eb = the_eb, .radius = 512}, {eq, shape1d},
      outlier_obj, &t_pred, stream);

  _2401::pszpred_lrz<T, E>::pszcxx_reverse_predict_lorenzo(
      {eq, shape1d}, {xdata, shape1d}, {.eb = the_eb, .radius = 512},
      {xdata, the_data_size}, &t_revpred, stream);

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