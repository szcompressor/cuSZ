#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>

#include "kernel/lrz/lrz.gpu.hh"
#include "utils/io.hh"

using std::string;
namespace utils = _portable::utils;

string fname;

template <typename T>
void profile_data_range(T* h_input, size_t const len, double& range)
{
  range = *std::max_element(h_input, h_input + len) - *std::min_element(h_input, h_input + len);
}

int main(int argc, char** argv)
{
  if (argc < 6) {
    printf(
        "0     1              2  3  4  5   6\n"
        "PROG  /path/to/data  X  Y  Z  eb  use_rel\n");
    exit(1);
  }
  else {
    fname = std::string(argv[1]);
    auto x = atoi(argv[2]);
    auto y = atoi(argv[3]);
    auto z = atoi(argv[4]);
    auto eb = std::stod(argv[5]);
    auto const use_rel = std::string(argv[6]) == "yes";

    auto len = x * y * z;

    using TIN = float;
    using TOUT = int16_t;

    TIN *h_in, *d_in;
    TOUT *h_out, *d_out;

    cudaMalloc(&d_in, len * sizeof(TIN));
    cudaMallocHost(&h_in, len * sizeof(TIN));

    cudaMalloc(&d_out, len * sizeof(TOUT));
    cudaMallocHost(&h_out, len * sizeof(TOUT));

    utils::fromfile(fname, h_in, len);

    printf("(x, y, z) = (%d, %d, %d)\n", x, y, z);

    if (use_rel) {
      printf("Using REL mode, input (REL) eb = %f, ", eb);
      double range;
      profile_data_range(h_in, len, range);
      eb *= range;
      printf("range = %f, adjusted (ABS) eb = %f.\n", range, eb);
    }
    else {
      printf("Using ABS mode, input (ABS) eb = %f, ", eb);
    }

    cudaMemcpy(d_in, h_in, len * sizeof(TIN), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float time_prequant;
    {
      psz::module::GPU_lorenzo_prequant<TIN, TOUT, false>(
          d_in, len, eb, d_out, &time_prequant, stream);
    }

    cudaMemcpy(h_out, d_out, len * sizeof(TOUT), cudaMemcpyDeviceToHost);
    utils::tofile(fname + ".prequant", h_out, len);

    cudaStreamDestroy(stream);
    cudaFree(d_in);
    cudaFreeHost(h_in);
  }

  return 0;
}