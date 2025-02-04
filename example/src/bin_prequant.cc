#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>

#include "kernel/lrz/lrz.gpu.hh"
#include "mem/cxx_backends.h"
#include "utils/io.hh"

using std::string;
namespace utils = _portable::utils;

template <typename T>
void profile_data_range(T* h_input, size_t const len, double& range)
{
  range = *std::max_element(h_input, h_input + len) - *std::min_element(h_input, h_input + len);
}

template <typename TIN = float, typename TOUT = int16_t>
int run(string fname, size_t x, size_t y, size_t z, double eb, bool use_rel)
{
  auto len = x * y * z;

  auto h_in = MAKE_UNIQUE_HOST(TIN, len);
  auto d_in = MAKE_UNIQUE_DEVICE(TIN, len);
  auto h_out = MAKE_UNIQUE_HOST(TOUT, len);
  auto d_out = MAKE_UNIQUE_DEVICE(TOUT, len);

  utils::fromfile(fname, h_in.get(), len);

  printf("(x, y, z) = (%lu, %lu, %lu)\n", x, y, z);

  if (use_rel) {
    printf("Using REL mode, input (REL) eb = %f, ", eb);
    double range;
    profile_data_range(h_in.get(), len, range);
    eb *= range;
    printf("range = %f, adjusted (ABS) eb = %f.\n", range, eb);
  }
  else {
    printf("Using ABS mode, input (ABS) eb = %f, ", eb);
  }

  auto ebx2 = eb * 2, ebx2_r = 1 / ebx2;

  memcpy_allkinds<H2D>(d_in.get(), h_in.get(), len);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  psz::module::GPU_lorenzo_prequant<TIN, TOUT, false>(
      d_in.get(), len, ebx2, ebx2_r, d_out.get(), stream);

  memcpy_allkinds<D2H>(h_out.get(), d_out.get(), len);
  utils::tofile(fname + ".prequant", h_out.get(), len);

  cudaStreamDestroy(stream);

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