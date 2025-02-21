#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>

#include "compbuf.hh"
#include "compressor.hh"
#include "cusz.h"
#include "kernel/lrz/lrz.gpu.hh"
#include "kernel/spv.hh"
#include "mem/cxx_backends.h"
#include "stat/compare.hh"
#include "utils/io.hh"

using std::string;
using namespace psz;
using _portable::utils::fromfile;
using psz::analysis::CPU_probe_extrema;
using psz::analysis::GPU_probe_extrema;

const int radius = 128;

template <typename T = float, typename TOUT = int16_t>
int run(string fname, size_t x, size_t y, size_t z, double eb, bool use_rel)
{
  auto len = x * y * z;

  auto h_origin = MAKE_UNIQUE_HOST(T, len);
  auto d_origin = MAKE_UNIQUE_DEVICE(T, len);
  auto h_reconst = MAKE_UNIQUE_HOST(T, len);
  auto d_reconst = MAKE_UNIQUE_DEVICE(T, len);

  fromfile(fname, h_origin.get(), len);

  printf("(x, y, z) = (%lu, %lu, %lu)\n", x, y, z);

  auto ebx2 = eb * 2, ebx2_r = 1 / ebx2;

  memcpy_allkinds<H2D>(d_origin.get(), h_origin.get(), len);

  if (use_rel) {
    printf("Using REL mode, input (REL) eb = %f, radius=%d, ", eb, radius);
    double _max_val, _min_val, range;
    {
      CPU_probe_extrema(h_origin.get(), len, _max_val, _min_val, range);
      printf("range (CPU profiled) = %f, ", range);
    }
    {
      GPU_probe_extrema(d_origin.get(), len, _max_val, _min_val, range);
      printf("range (GPU profiled) = %f.", range);
    }
    eb *= range;
    printf("\nrange = %f, adjusted (ABS) eb = %f.\n", range, eb);
  }
  else {
    printf("Using ABS mode, input (ABS) eb = %f, ", eb);
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  CompressorBufferToggle toggle{
      .err_ctrl_quant = true,
      .compact_outlier = true,
      .anchor = true,
      .histogram = false,
      .compressed = false,
  };

  CompressorBuffer<T> buf(x, y, z, true, &toggle);

  module::GPU_c_lorenzo_nd_with_outlier<T, false>(
      d_origin.get(), {x, y, z}, buf.ectrl(), (void*)buf.outlier(), buf.top1(), ebx2, ebx2_r,
      radius, stream);

  cudaStreamSynchronize(stream);

  float _;
  cout << "compact outliers: " << buf.compact_num_outliers() << endl;
  if (buf.compact_num_outliers() != 0)
    psz::spv_scatter_naive<CUDA>(
        buf.compact_val(), buf.compact_idx(), buf.compact_num_outliers(), d_reconst.get(), &_,
        stream);

  module::GPU_x_lorenzo_nd<T, false>(
      buf.ectrl(), d_reconst.get(), d_reconst.get(), {x, y, z}, ebx2, ebx2_r, radius, stream);

  cudaStreamDestroy(stream);

  auto s = new psz_statistics;
  psz::cuhip::GPU_assess_quality(s, d_origin.get(), d_reconst.get(), x * y * z);
  printf(
      "PSNR\t%lf\t"
      "NRMSE\t%lf\n",
      s->score_PSNR, s->score_NRMSE);

  return 0;
}

int main(int argc, char** argv)
{
  if (argc < 6) {
    printf(
        "0     1              2     3  4  5  6   7\n"
        "PROG  /path/to/data  dtype X  Y  Z  eb  use_rel\n");
    exit(1);
  }
  else {
    auto fname = std::string(argv[1]);
    auto dtype = std::string(argv[2]);
    auto x = atoi(argv[3]);
    auto y = atoi(argv[4]);
    auto z = atoi(argv[5]);
    auto eb = std::stod(argv[6]);
    auto const use_rel = std::string(argv[7]) == "yes";

    if (dtype == "f4" or dtype == "f32")
      return run<float, int16_t>(fname, x, y, z, eb, use_rel);
    else if (dtype == "f8" or dtype == "f64")
      return run<double, int16_t>(fname, x, y, z, eb, use_rel);
    else
      return -1;
  }
}