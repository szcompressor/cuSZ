#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "cusz.h"

namespace {

constexpr u4 X = 64, Y = 64, Z = 64;
constexpr size_t LEN = (size_t)X * Y * Z;

struct stage {
  char const* name;
  psz_predictor p1;
  psz_codec c1;
};

struct field {
  char const* name;
  std::vector<float> h;
  double eb;
};

long outside_bound(stage const& s, field const& f, float* d_in, float* d_out, void* stream)
{
  cudaMemcpy(d_in, f.h.data(), LEN * sizeof(float), cudaMemcpyHostToDevice);
  auto* enc = psz_compress_init(F4, psz_len{X, Y, Z}, stream);
  psz_header hdr{};
  u1* d_comp = nullptr;
  size_t comp_len = 0;
  long over = -1;
  if (psz_compress_process_float(enc, psz_ppl{s.p1, HistGeneric, s.c1, CodecNull}, f.eb, d_in) ==
          PSZ_SUCCESS and
      psz_compress_archive(enc, &hdr, &d_comp, &comp_len) == PSZ_SUCCESS) {
    auto* dec = psz_decompress_init(&hdr, stream);
    cudaMemset(d_out, 0xff, LEN * sizeof(float));
    if (psz_decompress_process_float(dec, d_comp, comp_len, d_out) == PSZ_SUCCESS) {
      cudaStreamSynchronize((cudaStream_t)stream);
      std::vector<float> back(LEN);
      cudaMemcpy(back.data(), d_out, LEN * sizeof(float), cudaMemcpyDeviceToHost);
      over = 0;
      for (size_t i = 0; i < LEN; i++)
        if (not(std::fabs((double)back[i] - f.h[i]) <= f.eb)) over++;
    }
    psz_free(dec);
  }
  psz_free(enc);
  return over;
}

}  // namespace

int main()
{
  std::vector<float> smooth(LEN), zeros(LEN, 0.0f);
  for (size_t i = 0; i < LEN; i++) {
    auto const x = i % X, y = (i / X) % Y, z = i / (X * Y);
    smooth[i] =
        0.5f + 0.4f * std::sin(x * 0.1f) * std::cos(y * 0.07f) * std::sin(z * 0.05f + 1.0f);
  }
  field const fields[] = {{"eb>range", smooth, 3.0}, {"zeros", zeros, 1e-3}};
  stage const stages[] = {
      {"lrz,hf", Lorenzo, HF_r2}, {"lrz,hfr-v2", Lorenzo, HFR}, {"spl,hf", SplineY25, HF_r2}};

  void* stream = nullptr;
  cudaStreamCreate((cudaStream_t*)&stream);
  float *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, LEN * sizeof(float));
  cudaMalloc(&d_out, LEN * sizeof(float));

  bool ok = true;
  for (auto const& f : fields)
    for (auto const& s : stages) {
      auto const over = outside_bound(s, f, d_in, d_out, stream);
      printf("  %-9s %-11s ", f.name, s.name);
      if (over == 0)
        printf("ok\n");
      else if (over < 0)
        printf("FAIL: compress or decompress returned an error\n");
      else
        printf("FAIL: %ld of %zu values outside eb=%g\n", over, LEN, f.eb);
      ok = ok and over == 0;
    }

  cudaFree(d_in);
  cudaFree(d_out);
  cudaStreamDestroy((cudaStream_t)stream);
  if (not ok) {
    printf("FAIL\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}
