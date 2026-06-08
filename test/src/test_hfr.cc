// Author: Jiannan Tian
// Test HFReVISIT alt-code encoding path without memobj.

#include <cuda_runtime.h>

#include "cusz/type.h"
#include "hf_impl.hh"
#include "phf.hh"
#include "utils/busyheader.hh"

template <typename T, int Magnitude = 12, int ReduceTimes = 4>
bool test_hfr_altcode(const char* label)
{
  using HFR = HFR_Config<T, Magnitude, ReduceTimes>;
  constexpr u4 bklen = 1024;
  constexpr size_t inlen = HFR::ChunkSize;
  constexpr size_t nblocks = 1;

  // codebook: all entries encode as 1-bit codeword "1"
  u4 *h_bk, *d_bk;
  cudaMallocHost(&h_bk, bklen * sizeof(u4));
  cudaMalloc(&d_bk, bklen * sizeof(u4));
  for (u4 i = 0; i < bklen; i++) h_bk[i] = HuffmanWord<4>(0b1, 1).to_uint();
  cudaMemcpy(d_bk, h_bk, bklen * sizeof(u4), cudaMemcpyHostToDevice);

  // derive alt-code from the host codebook
  u4 alt_code = 0, alt_bitcount = 0;
  phf::make_altcode_single(h_bk, (u2)bklen, ReduceTimes, alt_code, alt_bitcount);

  // input: all-zero quant codes (symbol 0)
  T* d_in;
  cudaMalloc(&d_in, inlen * sizeof(T));
  cudaMemset(d_in, 0, inlen * sizeof(T));

  // output buffers (post-breaks_t adoption)
  using BreakCell = psz::HFR_PBK_Breaks<128>;
  u4 *d_dn_out, *d_dn_bitcount, *h_dn_bitcount;
  BreakCell* d_sp_breaks;
  u4 *d_sp_count, *d_par_brnum, *d_par_broffset;
  uint8_t* d_par_encid;
  cudaMalloc(&d_dn_out, inlen * sizeof(u4));
  cudaMalloc(&d_dn_bitcount, nblocks * sizeof(u4));
  cudaMallocHost(&h_dn_bitcount, nblocks * sizeof(u4));
  cudaMalloc(&d_sp_breaks, inlen * sizeof(BreakCell));
  cudaMalloc(&d_sp_count, sizeof(u4));
  cudaMalloc(&d_par_brnum, nblocks * sizeof(u4));
  cudaMalloc(&d_par_broffset, nblocks * sizeof(u4));
  cudaMalloc(&d_par_encid, nblocks * sizeof(uint8_t));
  cudaMemset(d_sp_count, 0, sizeof(u4));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  phf::module::HFR_encoder<T, Magnitude, ReduceTimes, false, u4>::GPU_kernel_v1(
      d_in, inlen, d_bk, bklen, alt_code, alt_bitcount, d_dn_out, d_dn_bitcount, d_sp_breaks,
      d_sp_count, d_par_brnum, d_par_broffset, d_par_encid, stream);

  cudaStreamSynchronize(stream);

  cudaError_t err = cudaGetLastError();
  bool ok = (err == cudaSuccess);
  if (!ok) printf("[FAIL] CUDA error: %s\n", cudaGetErrorString(err));

  cudaMemcpy(h_dn_bitcount, d_dn_bitcount, nblocks * sizeof(u4), cudaMemcpyDeviceToHost);
  printf(
      "[%s] %s  M=%d RT=%d  alt_bitcount=%u  dn_bitcount[0]=%u\n", ok ? "PASS" : "FAIL", label,
      Magnitude, ReduceTimes, alt_bitcount, h_dn_bitcount[0]);

  cudaStreamDestroy(stream);
  cudaFreeHost(h_bk);
  cudaFree(d_bk);
  cudaFree(d_in);
  cudaFree(d_dn_out);
  cudaFree(d_dn_bitcount);
  cudaFreeHost(h_dn_bitcount);
  cudaFree(d_sp_breaks);
  cudaFree(d_sp_count);
  cudaFree(d_par_brnum);
  cudaFree(d_par_broffset);
  cudaFree(d_par_encid);

  return ok;
}

int main()
{
  auto ok = true;
  ok &= test_hfr_altcode<u2, 12, 4>("u2");
  ok &= test_hfr_altcode<u2, 12, 3>("u2");
  ok &= test_hfr_altcode<u2, 12, 2>("u2");
  return ok ? 0 : 1;
}
