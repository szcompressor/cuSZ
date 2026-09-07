// Correctness check: KCU_hfd26_fused decode vs. the original quant codes, exact element match.

#include <cstdio>
#include <memory>

#include "cxx_typing.h"
#include "hfd26.hh"
#include "mem/gpu_stream.hh"
#include "phf.hh"
#include "utils/synth.hh"

using E = u2;
using H = u4;
using Storage = u1;

extern "C" void* pbk25_r128_rvbk_d_ptr();

constexpr int DefaultReduceTimes = (int)psz::HFR_PBK_Constants::ReduceTimes;

// build an HFR-PBKGO archive, decode it via the fused path, compare against the original
template <int Mag, typename Eout = f4>
bool run_case(size_t len, int bklen, char const* synth_spec, char const* label)
{
  auto h_data = MAKE_UNIQUE_HOST(E, len);
  auto s = _ptb::testutils::Synth::parse(synth_spec);
  s.fill((void*)h_data.get(), len, _ptb::TypeSym<E>::type);

  size_t const d_alloc_len = ALIGN_4Ki(len);
  auto d_data = MAKE_UNIQUE_DEVICE(E, d_alloc_len);

  auto stream_owner = _ptb::make_gpu_stream();
  auto stream = stream_owner.get();

  if (d_alloc_len > len)
    cudaMemsetAsync(d_data.get() + len, 0, (d_alloc_len - len) * sizeof(E), (cudaStream_t)stream);
  memcpy_allkinds_async<H2D>(d_data.get(), h_data.get(), len, stream);
  sync_by_stream(stream);

  // HFR-PBKGO: prebuilt PBK25_R128 codebook; no histogram/book build needed.
  auto buf_enc = std::make_unique<phf::Buf<E>>(len, bklen, -1, /*use_HFR=*/true);

  u1* d_encoded = nullptr;
  size_t encoded_len = 0;
  phf_header header{};
  int rc = phf::high_level<E>::HFR_encode(
      buf_enc.get(), d_data.get(), len, &d_encoded, &encoded_len, header, stream,
      psz_codec::HFR_PBKGO, nullptr, nullptr, HFR_Opts{DefaultReduceTimes, Mag, 128});
  sync_by_stream(stream);
  if (rc != 0) {
    fprintf(stderr, "[%s] FAIL: HFR_encode returned %d\n", label, rc);
    return false;
  }

  auto bs_ptr = (H*)(d_encoded + header.entry[PHFHEADER_BITSTREAM]);
  auto packed_headers = (u4 const*)(d_encoded + header.entry[PHFHEADER_PBK_HEADERS]);
  size_t const bs_bytes = (size_t)header.total_ncell * sizeof(H);
  constexpr auto RvbkBytesPerBook = psz::HFR_PBK_Constants::RvbkBytesPerBook;
  auto rvbk_ptr = (u1*)pbk25_r128_rvbk_d_ptr();
  int const rvbk_bytes = (int)RvbkBytesPerBook;
  int const pardeg = (int)header.pardeg;

  auto d_decomp_fused = MAKE_UNIQUE_DEVICE(Eout, len);

  auto buf_fused = std::make_unique<phf::Buf<E>>(len, bklen, -1, /*use_HFR=*/true);
  phf::module::HFD26<E, H, Storage, Mag>::template decode_fused<Eout>(
      bs_ptr, bs_bytes, rvbk_ptr, rvbk_bytes, packed_headers, buf_fused->lut_d(), pardeg,
      header.ori_len, d_decomp_fused.get(), buf_fused->incomp_flag_d(), stream);

  sync_by_stream(stream);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "[%s] FAIL: CUDA error: %s\n", label, cudaGetErrorString(err));
    return false;
  }

  auto h_fused = MAKE_UNIQUE_HOST(Eout, len);
  memcpy_allkinds<D2H>(h_fused.get(), d_decomp_fused.get(), len);

  size_t mismatches = 0;
  size_t first_bad = (size_t)-1;
  for (size_t i = 0; i < len; i++) {
    if ((Eout)h_data[i] != h_fused[i]) {
      if (first_bad == (size_t)-1) first_bad = i;
      ++mismatches;
    }
  }

  if (mismatches > 0) {
    fprintf(
        stderr,
        "[%s] FAIL: %zu/%zu mismatches; first @ idx=%zu orig=%g fused=%g "
        "(pardeg=%d, chunk=%zu, off=%zu)\n",
        label, mismatches, len, first_bad, (double)h_data[first_bad], (double)h_fused[first_bad],
        pardeg, first_bad / (1u << Mag), first_bad % (1u << Mag));
    return false;
  }

  printf(
      "[%s] PASS: %zu elements identical (Mag=%d, pardeg=%d, bklen=%d)\n", label, len, Mag,
      pardeg, bklen);
  return true;
}

int main()
{
  bool ok = true;

  constexpr size_t Len = 6480000;  // matches the bin_hf ctest matrix scale
  int const bklen = 256;           // HFR-PBK family: radius=128 -> bklen 256

  // mild distribution: mostly dense LUT-decode chunks
  ok &= run_case<10>(Len, bklen, "cauchy:peak=128:gamma=0.254763:seed=43", "mag10/mild");

  // sharp distribution: exercises breaks/incomp/bypass chunks too
  ok &= run_case<10>(Len, bklen, "cauchy:peak=128:gamma=0.039351:seed=43", "mag10/sharp");

  // Mag=11: NumSegs=ShardsPerChunk/2 for the derive sub-phase (Option C boundary case)
  ok &= run_case<11>(Len, bklen, "cauchy:peak=128:gamma=0.254763:seed=43", "mag11/mild");
  ok &= run_case<11>(Len, bklen, "cauchy:peak=128:gamma=0.039351:seed=43", "mag11/sharp");

  // Mag=12: largest chunk size, NumSegs=ShardsPerChunk/2 too
  ok &= run_case<12>(Len, bklen, "cauchy:peak=128:gamma=0.254763:seed=43", "mag12/mild");
  ok &= run_case<12>(Len, bklen, "cauchy:peak=128:gamma=0.039351:seed=43", "mag12/sharp");

  printf(ok ? "[test_hfd26_fused] PASS\n" : "[test_hfd26_fused] FAIL\n");
  return ok ? 0 : 1;
}
