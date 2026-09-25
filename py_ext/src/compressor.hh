#ifndef PSZ_PYBIND_COMPRESSOR_HH
#define PSZ_PYBIND_COMPRESSOR_HH

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "cusz/header.h"
#include "cusz/type.h"

// cusz.h for nanobind: pointers and contexts cross as uintptr_t, OUT_ parameters come back.
namespace psz::pybind {

uintptr_t psz_compress_init(psz_dtype, size_t x, size_t y, size_t z, uintptr_t stream);
uintptr_t psz_compress_init_3stage(psz_dtype, size_t x, size_t y, size_t z, uintptr_t stream);
uintptr_t psz_decompress_init(psz_header const&, uintptr_t stream);
int psz_free(uintptr_t ctx);
int psz_last_error();

psz_data_summary psz_compress_extrema_float(uintptr_t ctx, uintptr_t d_in);
psz_data_summary psz_compress_extrema_double(uintptr_t ctx, uintptr_t d_in);

int psz_compress_process_float(uintptr_t ctx, psz_ppl, double eb, uintptr_t d_in);
int psz_compress_process_double(uintptr_t ctx, psz_ppl, double eb, uintptr_t d_in);

// (status, header, d_out, out_bytes); d_out points into the context's own buffer.
std::tuple<int, psz_header, uintptr_t, size_t> psz_compress_archive(uintptr_t ctx);
int psz_compress_reset(uintptr_t ctx);

int psz_decompress_process_float(uintptr_t ctx, uintptr_t d_in, size_t in_bytes, uintptr_t out);
int psz_decompress_process_double(uintptr_t ctx, uintptr_t d_in, size_t in_bytes, uintptr_t out);
int psz_decompress_reset(uintptr_t ctx);

// The psz_stats fields the CUDA assessor fills.
struct Quality {
  double psnr, mse, nrmse, coeff;
  double max_err_abs, max_err_rel, max_err_pwrrel;
  size_t max_err_idx;
  double origin_min, origin_max, origin_rng, origin_std, origin_avg;
  size_t len;
};
std::tuple<int, Quality> psz_assess_quality_float(
    uintptr_t d_reconst, uintptr_t d_origin, size_t len);
std::tuple<int, Quality> psz_assess_quality_double(
    uintptr_t d_reconst, uintptr_t d_origin, size_t len);

void psz_print_concise_quality(psz_header const&, Quality const&, size_t comp_bytes);

void psz_review_compression(psz_header const&);
void psz_review_compression_verbose(psz_header const&);
void psz_review_decompression(psz_header const&);
void psz_review_decompression_verbose(psz_header const&);

// Not in cusz.h: the psz_header at the front of an archive, one device-to-host copy.
psz_header header_from_archive(uintptr_t d_archive);

}  // namespace psz::pybind

#endif /* PSZ_PYBIND_COMPRESSOR_HH */
