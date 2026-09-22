// One ctx, many compressions: a reused buf must not carry state between runs.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "cusz_rev1.h"

namespace {

constexpr int NITER = 4;
constexpr size_t LEN = (size_t)128 * 128 * 64;

struct shape {
  char const* name;
  u4 x, y, z;
};

struct stage {
  char const* name;
  psz_predictor p1;
  psz_codec c1, c2;
  bool needs_3d = false;
};

std::vector<float> smooth_field()
{
  std::vector<float> h(LEN);
  for (size_t i = 0; i < LEN; i++) {
    auto const x = i % 128, y = (i / 128) % 128, z = i / (128 * 128);
    h[i] = std::sin(x * 0.05f) * std::cos(y * 0.04f) + 0.3f * std::sin(z * 0.07f) +
           0.01f * ((x * 7 + y * 13 + z * 29) % 11);
  }
  return h;
}

bool run(stage const& s, shape const& sh, float* d_in, float* d_out, void* stream)
{
  if (s.needs_3d and sh.z == 1) return true;
  printf("  %-8s %-14s ... ", sh.name, s.name), fflush(stdout);
  auto* enc = psz_init_from_stages(F4, psz_len{sh.x, sh.y, sh.z}, s.p1, s.c1, s.c2, stream);
  if (not enc) {
    printf("SKIP\n");
    return true;
  }

  size_t len0 = 0;
  unsigned long long sum0 = 0;
  bool ok = true;
  psz_ctx* dec = nullptr;
  std::vector<float> back(LEN);

  for (int it = 0; it < NITER; it++) {
    psz_header hdr{};
    u1* d_comp = nullptr;
    size_t comp_len = 0;
    psz_rc2 rc{Rel, 1e-3};

    if (psz_compress_float(enc, rc, d_in, &hdr, &d_comp, &comp_len) != PSZ_SUCCESS) {
      printf("  %-8s %-14s FAIL compress at iter %d\n", sh.name, s.name, it);
      ok = false;
      break;
    }
    if (not dec) dec = psz_init_from_header(&hdr, stream);

    cudaMemsetAsync(d_out, 0, LEN * sizeof(float), (cudaStream_t)stream);
    if (psz_decompress_float(dec, d_comp, comp_len, d_out) != PSZ_SUCCESS) {
      printf("  %-8s %-14s FAIL decompress at iter %d\n", sh.name, s.name, it);
      ok = false;
      break;
    }
    cudaStreamSynchronize((cudaStream_t)stream);
    cudaMemcpy(back.data(), d_out, LEN * sizeof(float), cudaMemcpyDeviceToHost);

    unsigned long long sum = 0;
    for (size_t i = 0; i < LEN; i++) {
      u4 w;
      memcpy(&w, &back[i], 4);
      sum = sum * 1000003ull + w;
    }

    if (it == 0) { len0 = comp_len, sum0 = sum; }
    else {
      if (comp_len != len0) {
        printf("  %-8s %-14s FAIL iter %d: %zu B, iter 0 gave %zu B\n", sh.name, s.name, it, comp_len, len0);
        ok = false;
      }
      if (sum != sum0) {
        printf("  %-8s %-14s FAIL iter %d: reconstruction differs\n", sh.name, s.name, it);
        ok = false;
      }
    }
  }
  if (ok) printf("ok  (%d iters, %zu B each)\n", NITER, len0);
  if (dec) psz_free(dec);
  psz_free(enc);
  return ok;
}

}  // namespace

int main()
{
  stage const stages[] = {
      {"lrz,hf", Lorenzo, HF_r2, CodecNull},
      {"lrz,hfr-v2", Lorenzo, HFR, CodecNull},
      {"lrz,hfr-v4", Lorenzo, HFR_V4, CodecNull},
      {"lrz,hfr-pbkc", Lorenzo, HFR_PBKC, CodecNull},
      {"lrz,hfr-pbkgo", Lorenzo, HFR_PBKGO, CodecNull},
      {"lrz-zz,fzg", LorenzoZigZag, FZG, CodecNull},
      {"spl-y24,hf", SplineY24, HF_r2, CodecNull, true},  // 3D only (context.cc:158)
      {"spl-y25,hf", SplineY25, HF_r2, CodecNull, true},
      {"lrz,lc-drh", Lorenzo, LC_DRH, CodecNull},
  };

  shape const shapes[] = {{"3D", 128, 128, 64}, {"2D", 1024, 1024, 1}, {"1D", 1048576, 1, 1}};

  void* stream = nullptr;
  cudaStreamCreate((cudaStream_t*)&stream);

  auto h = smooth_field();
  float *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, LEN * sizeof(float));
  cudaMalloc(&d_out, LEN * sizeof(float));
  cudaMemcpy(d_in, h.data(), LEN * sizeof(float), cudaMemcpyHostToDevice);

  bool ok = true;
  for (auto const& sh : shapes)
    for (auto const& s : stages) ok &= run(s, sh, d_in, d_out, stream);

  cudaFree(d_in), cudaFree(d_out);
  cudaStreamDestroy((cudaStream_t)stream);

  printf("test_iterative_reuse: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
