// Byte-stream regression sweep for psz::cuda::GPU_identical (sizeof_T \in {1,2,4}, sizes
// straddling the 262144-byte threshold, perturb at head/mid/tail).
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

#include "compare.hh"
#include "mem/cxx_backends.h"
#include "mem/cxx_smart_ptr.h"

namespace {

struct Case {
  char const* name;
  size_t len;
  size_t sizeof_T;
  long long perturb_byte;
  bool expect_identical;
};

bool run_case(Case const& c, cudaStream_t stream)
{
  size_t const bytes = c.len * c.sizeof_T;

  auto h1 = MAKE_UNIQUE_HOST(uint8_t, bytes);
  auto h2 = MAKE_UNIQUE_HOST(uint8_t, bytes);
  auto d1 = MAKE_UNIQUE_DEVICE(uint8_t, bytes);
  auto d2 = MAKE_UNIQUE_DEVICE(uint8_t, bytes);

  std::mt19937 rng(0xC0DECAFEu ^ (uint32_t)bytes);
  for (size_t i = 0; i < bytes; ++i) h1.get()[i] = (uint8_t)(rng() & 0xff);
  std::memcpy(h2.get(), h1.get(), bytes);
  if (c.perturb_byte >= 0) h2.get()[c.perturb_byte] ^= 0x5A;

  memcpy_allkinds<H2D>(d1.get(), h1.get(), bytes);
  memcpy_allkinds<H2D>(d2.get(), h2.get(), bytes);

  bool got_cpu = psz::cppstl::CPU_identical(h1.get(), h2.get(), c.sizeof_T, c.len);
  bool got_gpu = psz::cuda::GPU_identical(d1.get(), d2.get(), c.sizeof_T, c.len, stream);

  bool ok = (got_cpu == c.expect_identical) and (got_gpu == c.expect_identical);
  std::printf(
      "  [%s] %-28s len=%zu sizeof_T=%zu perturb=%lld  cpu=%d gpu=%d expected=%d\n",
      ok ? "PASS" : "FAIL", c.name, c.len, c.sizeof_T, c.perturb_byte, (int)got_cpu, (int)got_gpu,
      (int)c.expect_identical);
  return ok;
}

}  // namespace

int main()
{
  Case const cases[] = {
      {"id-small-u1", 1027, 1, -1, true},
      {"id-small-u2", 1027, 2, -1, true},
      {"id-small-u4", 1027, 4, -1, true},
      {"id-large-u1", 1ull << 20, 1, -1, true},
      {"id-large-u2", 6'480'000ull, 2, -1, true},
      {"id-large-u4", 1ull << 20, 4, -1, true},
      {"diff-small-tail", 1027, 4, 4 * 1025, false},
      {"diff-large-u1-head", 1ull << 20, 1, 0, false},
      {"diff-large-u1-mid", 1ull << 20, 1, 1ull << 19, false},
      {"diff-large-u1-tail", 1ull << 20, 1, (1ull << 20) - 1, false},
      {"diff-large-u2-mid", 6'480'000ull, 2, 6'480'000ull, false},
      {"diff-large-u2-tail", 6'480'000ull, 2, 6'480'000ull * 2 - 1, false},
      {"diff-large-u4-mid", 1ull << 20, 4, 1ull << 21, false},
  };

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int n_pass = 0, n_total = (int)(sizeof(cases) / sizeof(cases[0]));
  for (auto const& c : cases) n_pass += run_case(c, stream) ? 1 : 0;

  cudaStreamDestroy(stream);

  std::printf("\n%d / %d passed\n", n_pass, n_total);
  return (n_pass == n_total) ? 0 : 1;
}
