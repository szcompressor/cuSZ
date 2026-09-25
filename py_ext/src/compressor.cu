#include <cuda_runtime.h>

#include <cstdio>

#include "compressor.hh"
#include "cusz.h"

namespace psz::pybind {

uintptr_t psz_compress_init(psz_dtype dtype, size_t x, size_t y, size_t z, uintptr_t stream)
{
  return reinterpret_cast<uintptr_t>(
      ::psz_compress_init(dtype, psz_len{x, y, z}, reinterpret_cast<void*>(stream)));
}

uintptr_t psz_compress_init_3stage(psz_dtype dtype, size_t x, size_t y, size_t z, uintptr_t stream)
{
  return reinterpret_cast<uintptr_t>(
      ::psz_compress_init_3stage(dtype, psz_len{x, y, z}, reinterpret_cast<void*>(stream)));
}

uintptr_t psz_decompress_init(psz_header const& header, uintptr_t stream)
{
  psz_header h = header;  // psz_decompress_init takes a non-const pointer
  return reinterpret_cast<uintptr_t>(::psz_decompress_init(&h, reinterpret_cast<void*>(stream)));
}

int psz_free(uintptr_t ctx) { return ::psz_free(reinterpret_cast<psz_ctx*>(ctx)); }
int psz_last_error() { return ::psz_last_error(); }

#define PSZ_PYBIND_EXTREMA_IMPL(SUFFIX, T)                                      \
  psz_data_summary psz_compress_extrema_##SUFFIX(uintptr_t ctx, uintptr_t d_in) \
  {                                                                             \
    return ::psz_compress_extrema_##SUFFIX(                                     \
        reinterpret_cast<psz_ctx*>(ctx), reinterpret_cast<T*>(d_in));           \
  }

PSZ_PYBIND_EXTREMA_IMPL(float, float)
PSZ_PYBIND_EXTREMA_IMPL(double, double)
#undef PSZ_PYBIND_EXTREMA_IMPL

#define PSZ_PYBIND_COMPRESS_IMPL(SUFFIX, T)                                                     \
  int psz_compress_process_##SUFFIX(uintptr_t ctx, psz_ppl pipeline, double eb, uintptr_t d_in) \
  {                                                                                             \
    return ::psz_compress_process_##SUFFIX(                                                     \
        reinterpret_cast<psz_ctx*>(ctx), pipeline, eb, reinterpret_cast<T*>(d_in));             \
  }

PSZ_PYBIND_COMPRESS_IMPL(float, float)
PSZ_PYBIND_COMPRESS_IMPL(double, double)
#undef PSZ_PYBIND_COMPRESS_IMPL

std::tuple<int, psz_header, uintptr_t, size_t> psz_compress_archive(uintptr_t ctx)
{
  psz_header header{};
  uint8_t* d_out = nullptr;
  size_t out_bytes = 0;
  auto stat = ::psz_compress_archive(reinterpret_cast<psz_ctx*>(ctx), &header, &d_out, &out_bytes);
  return {stat, header, reinterpret_cast<uintptr_t>(d_out), out_bytes};
}

int psz_compress_reset(uintptr_t ctx)
{ return ::psz_compress_reset(reinterpret_cast<psz_ctx*>(ctx)); }

#define PSZ_PYBIND_DECOMPRESS_IMPL(SUFFIX, T)                                        \
  int psz_decompress_process_##SUFFIX(                                               \
      uintptr_t ctx, uintptr_t d_in, size_t in_bytes, uintptr_t out)                 \
  {                                                                                  \
    return ::psz_decompress_process_##SUFFIX(                                        \
        reinterpret_cast<psz_ctx*>(ctx), reinterpret_cast<uint8_t*>(d_in), in_bytes, \
        reinterpret_cast<T*>(out));                                                  \
  }

PSZ_PYBIND_DECOMPRESS_IMPL(float, float)
PSZ_PYBIND_DECOMPRESS_IMPL(double, double)
#undef PSZ_PYBIND_DECOMPRESS_IMPL

int psz_decompress_reset(uintptr_t ctx)
{ return ::psz_decompress_reset(reinterpret_cast<psz_ctx*>(ctx)); }

#define PSZ_PYBIND_QUALITY_IMPL(SUFFIX, T)                                                  \
  std::tuple<int, Quality> psz_assess_quality_##SUFFIX(                                     \
      uintptr_t d_reconst, uintptr_t d_origin, size_t len)                                  \
  {                                                                                         \
    psz_stats s{};                                                                          \
    auto stat = ::psz_assess_quality_##SUFFIX(                                              \
        &s, reinterpret_cast<T*>(d_reconst), reinterpret_cast<T*>(d_origin), len);          \
    return {                                                                                \
        stat, Quality{                                                                      \
                  s.score.PSNR, s.score.MSE, s.score.NRMSE, s.score.coeff, s.max_err.abs,   \
                  s.max_err.rel, s.max_err.pwrrel, s.max_err.idx, s.odata.min, s.odata.max, \
                  s.odata.rng, s.odata.std, s.odata.avg, s.len}};                           \
  }

PSZ_PYBIND_QUALITY_IMPL(float, float)
PSZ_PYBIND_QUALITY_IMPL(double, double)
#undef PSZ_PYBIND_QUALITY_IMPL

void psz_print_concise_quality(psz_header const& header, Quality const& q, size_t comp_bytes)
{
  psz_stats s{};
  s.score = {q.psnr, q.mse, q.nrmse, q.coeff};
  s.max_err = {q.max_err_abs, q.max_err_rel, q.max_err_pwrrel, q.max_err_idx};
  s.odata = {q.origin_min, q.origin_max, q.origin_rng, q.origin_std, q.origin_avg};
  s.len = q.len;
  ::psz_print_concise_quality(const_cast<psz_header*>(&header), &s, comp_bytes);
  fflush(stdout);
}

void psz_review_compression(psz_header const& header)
{
  ::psz_review_compression(const_cast<psz_header*>(&header));
}
void psz_review_compression_verbose(psz_header const& header)
{
  ::psz_review_compression_verbose(const_cast<psz_header*>(&header));
}
void psz_review_decompression(psz_header const& header)
{
  ::psz_review_decompression(const_cast<psz_header*>(&header));
}
void psz_review_decompression_verbose(psz_header const& header)
{
  ::psz_review_decompression_verbose(const_cast<psz_header*>(&header));
}

psz_header header_from_archive(uintptr_t d_archive)
{
  psz_header h{};
  cudaMemcpy(&h, reinterpret_cast<void*>(d_archive), sizeof(psz_header), cudaMemcpyDeviceToHost);
  return h;
}

}  // namespace psz::pybind
