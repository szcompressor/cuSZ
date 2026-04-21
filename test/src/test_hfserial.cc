/**
 * @file test_hfserial.cc
 * @author Jiannan Tian
 * @brief Integration test for CPU serial Huffman codebook construction.
 *        Revived from example/src/bin_hfserial.cc without memobj.
 *        Standalone: generates a synthetic histogram.
 *        File mode (args: <hist-bin> <bklen>): reads histogram from binary.
 * @version 0.5
 * @date 2024-04-20
 *
 * (C) 2024 by Indiana University, Argonne National Laboratory
 *
 */

#include "cusz/type.h"
#include "hf_impl.hh"
#include "utils/busyheader.hh"

static void print_code_u4(u4 idx, u4 word)
{
  using PW = HuffmanWord<4>;
  auto pw = reinterpret_cast<PW*>(&word);
  cout << idx << "\t"
       << bitset<PW::FIELD_BITCOUNT>(pw->bitcount) << " (" << pw->bitcount << ")\t"
       << bitset<PW::FIELD_CODE>(pw->prefix_code) << "\n";
}

// Build codebook with phf_CPU_build_canonized_codebook_v2 and verify validity.
static bool test_hf_cpu_codebook(u4* freq, int bklen, bool verbose = false)
{
  auto revbook_bytes = hf_space<u4, u4>::revbook_bytes(bklen);
  auto bk4 = new u4[bklen];
  auto revbook = new u1[revbook_bytes];
  memset(bk4, 0xff, sizeof(u4) * bklen);
  memset(revbook, 0, revbook_bytes);

  phf_CPU_build_canonized_codebook_v2<u4, u4>(freq, bklen, bk4, revbook, revbook_bytes);

  auto ok = true;
  for (auto i = 0; i < bklen; i++) {
    if (freq[i] == 0) continue;
    auto pw = reinterpret_cast<HuffmanWord<4>*>(bk4 + i);
    if (pw->bitcount == 0 || pw->bitcount > HuffmanWord<4>::FIELD_CODE) {
      printf("[FAIL] symbol %d: freq=%u but invalid bitcount=%u\n", i, freq[i], pw->bitcount);
      ok = false;
    }
  }

  if (verbose) {
    printf("codebook (non-trivial entries):\n");
    for (auto i = 0; i < bklen; i++) {
      if (bk4[i] != 0 && bk4[i] != 0xffffffff) print_code_u4(i, bk4[i]);
    }
  }

  delete[] bk4;
  delete[] revbook;
  return ok;
}

// Gaussian-like histogram centered at bklen/2.
static u4* make_synthetic_hist(int bklen, int inlen = 1 << 20)
{
  auto hist = new u4[bklen]();
  auto center = bklen / 2;
  auto sigma2 = (double)bklen * bklen / 8.0;
  for (auto i = 0; i < bklen; i++) {
    auto d = (double)(i - center);
    hist[i] = (u4)(inlen * exp(-d * d / sigma2));
  }
  if (hist[center] == 0) hist[center] = inlen;
  return hist;
}

int main(int argc, char** argv)
{
  int bklen = 1024;
  u4* freq = nullptr;

  if (argc >= 3) {
    bklen = atoi(argv[2]);
    freq = new u4[bklen]();
    auto f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    auto nread = fread(freq, sizeof(u4), bklen, f);
    if ((int)nread != bklen) { fprintf(stderr, "short read: %zu of %d\n", nread, bklen); fclose(f); return 1; }
    fclose(f);
  }
  else {
    freq = make_synthetic_hist(bklen);
  }

  auto ok = test_hf_cpu_codebook(freq, bklen, argc < 3);
  printf("[%s] CPU serial HF codebook (bklen=%d)\n", ok ? "PASS" : "FAIL", bklen);

  delete[] freq;
  return ok ? 0 : 1;
}
