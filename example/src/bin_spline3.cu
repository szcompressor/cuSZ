/**
 * @file bin_spline3.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-06-06
 * (create) 2021-06-06 (rev.1) 2022-01-08 (rev.2) 2023-08-04
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include <iostream>

#include "kernel/spline.hh"
#include "kernel/histsp.hh"
#include "mem/memseg_cxx.hh"
#include "spline3_driver.cuh"
#include "stat/compare_gpu.hh"
#include "stat/stat.hh"
#include "utils/cuda_err.cuh"
#include "utils/print_gpu.hh"
#include "utils/timer.hh"
#include "utils/viewer.hh"

using std::cout;

using T = float;
using E = float;  // 23-08-08 must be float in avoidance of decomp err

using Predictor = cusz::Spline3<T, E>;
// double const  eb          = 3e-3;
constexpr int fake_radius = 0;

constexpr uint32_t dimz = 449, dimy = 449, dimx = 235;
constexpr dim3 xyz = dim3(dimx, dimy, dimz);
constexpr uint32_t len = dimx * dimy * dimz;

double _1, _2, rng;

std::string fname("");

template <typename CAPSULE>
void adjust_eb(CAPSULE& data, double& eb, double& adjusted_eb, bool use_r2r)
{
  adjusted_eb = eb;

  if (use_r2r) {
    printf("using r2r mode...\n");
    adjusted_eb *= rng;
    printf("rng: %f\teb: %f\tadjusted eb: %f\n", rng, eb, adjusted_eb);
  }
  else {
    printf("using abs mode...\n");
    printf("eb: %f\n", eb);
  }
}

void predictor_detail(
    T* d_data, T* d_cmp, dim3 xyz, double eb, bool use_sp,
    cudaStream_t stream = nullptr)
{
  Predictor predictor(xyz, true);

  T* d_xdata = d_data;

  printf("allocate...\n");
  predictor.allocate();

  printf("compress...\n");
  predictor.construct(
      d_data, predictor.data_size, predictor.data_leap, eb, fake_radius,
      predictor.ectrl->dptr(), predictor.size_aligned, predictor.leap_aligned,
      predictor.anchor->dptr(), predictor.anchor_leap, stream);

  printf("decompress...\n");
  predictor.reconstruct(
      predictor.ectrl->dptr(), predictor.size_aligned, predictor.leap_aligned,
      predictor.anchor->dptr(), predictor.anchor_size, predictor.anchor_leap,
      eb, fake_radius, d_xdata, predictor.data_size, predictor.data_leap,
      stream);

  cusz::QualityViewer::echo_metric_gpu(d_xdata, d_cmp, len);
}

void predictor_demo(
    bool use_sp, double eb = 1e-2, bool use_compressor = false,
    bool use_r2r = false)
{
  auto exp = new pszmem_cxx<T>(len, 1, 1, "exp data");
  auto bak = new pszmem_cxx<T>(len, 1, 1, "bak data");

  cout << "using eb = " << eb << '\n';
  cout << fname << '\n';

  exp->control({Malloc, MallocHost})
      ->file(fname.c_str(), FromFile)
      ->control({H2D})
      ->extrema_scan(_1, _2, rng);
  bak->control({Malloc, MallocHost});

  cudaMemcpy(bak->hptr(), exp->hptr(), len * sizeof(T), cudaMemcpyHostToHost);
  bak->control({H2D});

  double adjusted_eb;
  adjust_eb(exp, eb, adjusted_eb, use_r2r);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  predictor_detail(exp->dptr(), bak->dptr(), xyz, adjusted_eb, use_sp, stream);

  CHECK_CUDA(cudaStreamDestroy(stream));

  exp->control({Free, FreeHost});
  bak->control({Free, FreeHost});
}

int main(int argc, char** argv)
{
  auto help = []() {
    cout << "        1      2   3      \n";
    cout << "./prog  fname  eb  abs|r2r\n";
  };

  auto eb = 1e-2;
  auto mode = std::string("abs");
  auto use_r2r = false;

  if (argc < 4) {  //
    help();
  }
  else {
    fname = std::string(argv[1]);
    eb = atof(argv[2]);
    mode = std::string(argv[3]);
    use_r2r = mode == "r2r";

    predictor_demo(false, eb, false, use_r2r);
  }

  return 0;
}
