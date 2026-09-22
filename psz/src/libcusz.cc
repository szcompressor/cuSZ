#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "compare.hh"
#include "compressor.hh"
#include "context_impl.h"
#include "cusz.h"
#include "extrema.hh"
#include "mem/buf_comp.hh"
#include "module.hh"

using psz::_2609::compose;
using psz::_2609::needs_eq4;
using std::cerr;
using std::endl;
template <typename T, typename E>
using CP = psz::compression_pipeline<T, E>;

static thread_local psz_error_status last_error = PSZ_SUCCESS;

psz_error_status psz_last_error() { return last_error; }

const char* psz_error_string(int e)
{
  switch (e) {
    case PSZ_SUCCESS: return "success";
    case PSZ_WARN_RADIUS_TOO_LARGE: return "radius too large";
    case PSZ_WARN_OUTLIER_TOO_MANY: return "more outliers than the outlier buffer holds";
    case PSZ_ABORT_UNSUPPORTED_TYPE: return "unsupported data type";
    case PSZ_ABORT_UNSUPPORTED_DIMENSION: return "unsupported dimensionality";
    case PSZ_ABORT_NOT_IMPLEMENTED: return "not implemented";
    case PSZ_ABORT_NO_SUCH_PREDICTOR: return "no such predictor";
    case PSZ_ABORT_NO_SUCH_CODEC: return "no such codec";
    case PSZ_ABORT_TOO_MANY_UNPREDICTABLE: return "too many unpredictables";
    case PSZ_ABORT_TOO_MANY_ENC_BREAK: return "too many encoding breaks";
    case PSZ_ABORT_COMPRESSED_TOO_LARGE: return "compressed size exceeds the output buffer";
    case PSZ_ABORT_UNSUPPORTED_PIPELINE: return "unsupported pipeline";
    default: return "unknown error";
  }
}

static psz_ctx* fail(psz_error_status status)
{
  last_error = status;
  return nullptr;
}

static bool pick_eq4(psz_ctx* m)
{
  m->use_eq4 = needs_eq4(m->header->pipeline);
  return m->buf != nullptr;
}

static psz_ctx* make_manager(
    psz_dtype dtype, psz_len len, psz_ppl pipeline, void* stream, uint8_t* d_archive = nullptr,
    size_t archive_capacity = 0)
{
  if (not psz::_2609::valid(pipeline)) return fail(PSZ_ABORT_UNSUPPORTED_PIPELINE);

  auto m = new psz_ctx;
  m->d_archive = d_archive;
  m->archive_capacity = archive_capacity;

  auto defaults = pszctx_default_values();
  m->header = new psz_header();
  memcpy(m->header, defaults->header, sizeof(psz_header));
  delete defaults;

  m->header->dtype = dtype;
  m->header->pipeline = pipeline;
  m->header->len = len;
  m->len_linear = len.x * len.y * len.z;
  m->header->radius = psz::_2609::radius_of(pipeline);  // HFR books assume 128
  m->bklen = m->header->radius * 2;
  m->cli = nullptr;
  m->use_eq4 = needs_eq4(pipeline);

  int const nstage = psz::_2609::nstage_of(pipeline);
  size_t const archive_eq4 = dtype == F4
                                 ? psz::Buf_Comp<f4>::compressed_max_bytes(len, nstage, true)
                                 : psz::Buf_Comp<f8>::compressed_max_bytes(len, nstage, true);
  bool const fits_eq4 = not d_archive or archive_capacity >= archive_eq4;
  m->buf = dtype == F4 ? CP<f4, u2>::compress_init(m, d_archive, nstage, fits_eq4)
                       : CP<f8, u2>::compress_init(m, d_archive, nstage, fits_eq4);
  m->stream = stream;

  last_error = PSZ_SUCCESS;
  return m;
}

psz_ctx* psz_init(psz_dtype dtype, psz_len len, psz_ppl pipeline, void* stream)
{ return make_manager(dtype, len, pipeline, stream); }

size_t psz_archive_capacity(
    psz_dtype dtype, psz_len len, psz_predictor p1, psz_codec c1, psz_codec optional_c2)
{
  auto const ppl = compose(p1, c1, optional_c2);
  auto const nstage = psz::_2609::nstage_of(ppl);
  return dtype == F4 ? psz::Buf_Comp<f4>::compressed_max_bytes(len, nstage, needs_eq4(ppl))
                     : psz::Buf_Comp<f8>::compressed_max_bytes(len, nstage, needs_eq4(ppl));
}

psz_ctx* psz_init_with_archive(
    psz_dtype dtype, psz_len len, psz_predictor p1, psz_codec c1, psz_codec optional_c2,
    uint8_t* d_archive, size_t archive_capacity, void* stream)
{
  if (d_archive and
      archive_capacity < psz_archive_capacity(dtype, len, p1, c1, optional_c2))
    return fail(PSZ_ABORT_UNSUPPORTED_PIPELINE);
  return make_manager(
      dtype, len, compose(p1, c1, optional_c2), stream, d_archive, archive_capacity);
}

psz_ctx* psz_init_from_stages(
    psz_dtype dtype, psz_len len, psz_predictor p1, psz_codec c1, psz_codec optional_c2,
    void* stream)
{ return make_manager(dtype, len, compose(p1, c1, optional_c2), stream); }

psz_ctx* psz_init_from_preset(psz_dtype dtype, psz_len len, psz_preset preset, void* stream)
{
  if (psz::_2609::is_generic(preset)) return fail(PSZ_ABORT_UNSUPPORTED_PIPELINE);

  return make_manager(dtype, len, psz::_2609::pipeline_of(preset), stream);
}

psz_ctx* psz_init_from_header(psz_header* header, void* stream)
{
  if (not psz::_2609::valid(header->pipeline)) return fail(PSZ_ABORT_UNSUPPORTED_PIPELINE);

  auto m = new psz_ctx;
  last_error = PSZ_SUCCESS;
  m->header = new psz_header();
  memcpy(m->header, header, sizeof(psz_header));
  m->bklen = m->header->radius * 2;
  m->len_linear = header->len.x * header->len.y * header->len.z;
  m->cli = nullptr;
  m->use_eq4 = needs_eq4(header->pipeline);

  int const nstage = psz::_2609::nstage_of(header->pipeline);
  m->buf = header->dtype == F4 ? CP<f4, u2>::decompress_init(m->header, nstage)
                               : CP<f8, u2>::decompress_init(m->header, nstage);

  m->stream = stream;

  return m;
}

int psz_free(psz_ctx* manager)
{
  auto dtype = manager->header->dtype;
  if (dtype == F4)
    delete (psz::Buf_Comp<f4>*)manager->buf;
  else if (dtype == F8)
    delete (psz::Buf_Comp<f8>*)manager->buf;
  else
    return PSZ_ABORT_UNSUPPORTED_TYPE;

  if (manager->cli) delete manager->cli;
  if (manager->header) delete manager->header;
  delete manager;

  return 0;
}

#define RUNTIME_SAVE_CONFIG2()                                \
  if (not pick_eq4(m)) return PSZ_ABORT_UNSUPPORTED_PIPELINE; \
  m->header->eb = rc.eb;                                      \
  m->header->user_input_eb = rc.eb;                           \
  m->header->radius = psz::_2609::radius_of(m->header->pipeline);     \
  m->bklen = m->header->radius * 2;

#define RUNTIME_CHANGE_EB_IF_REL(Type)                                      \
  if (rc.mode == Rel) {                                                     \
    auto [min_val, max_val, avg_val, rng] =                                 \
        psz::cuda::GPU_get_extrema<Type>::kernel(IN_d_data, m->len_linear); \
    (void)avg_val;                                                          \
    m->header->min_val = min_val;                                           \
    m->header->max_val = max_val;                                           \
    m->header->eb *= rng;                                                   \
  }

int psz_compress_float(
    psz_ctx* m, psz_rc2 rc, float* IN_d_data, psz_header* OUT_header, uint8_t** OUT_d_compressed,
    size_t* OUT_compressed_bytes)
{
  int status = PSZ_SUCCESS;

  RUNTIME_SAVE_CONFIG2();
  RUNTIME_CHANGE_EB_IF_REL(float);

  if (m->use_eq4) {
    status = CP<f4, u4>::compress(
        m, (psz_buf<f4>*)m->buf, IN_d_data, OUT_d_compressed, OUT_compressed_bytes, m->stream);
    ((psz_buf<f4>*)m->buf)->reset(m->stream);
    if (status != PSZ_SUCCESS) return status;
    *OUT_header = *(m->header);
    memcpy_allkinds<H2D>((u1*)*OUT_d_compressed, (u1*)m->header, sizeof(psz_header));
    if (m->cli) CP<f4, u4>::compress_dump_internal_buf(m, (psz_buf<f4>*)m->buf, m->stream);
  }
  else {
    status = CP<f4, u2>::compress(
        m, (psz_buf<f4>*)m->buf, IN_d_data, OUT_d_compressed, OUT_compressed_bytes, m->stream);
    ((psz_buf<f4>*)m->buf)->reset(m->stream);
    if (status != PSZ_SUCCESS) return status;
    *OUT_header = *(m->header);
    memcpy_allkinds<H2D>((u1*)*OUT_d_compressed, (u1*)m->header, sizeof(psz_header));
    if (m->cli) CP<f4, u2>::compress_dump_internal_buf(m, (psz_buf<f4>*)m->buf, m->stream);
  }

  return status;
}

int psz_compress_double(
    psz_ctx* m, psz_rc2 rc, double* IN_d_data, psz_header* OUT_header, uint8_t** OUT_d_compressed,
    size_t* OUT_compressed_bytes)
{
  int status = PSZ_SUCCESS;

  RUNTIME_SAVE_CONFIG2();
  RUNTIME_CHANGE_EB_IF_REL(double);

  if (m->use_eq4) {
    status = CP<f8, u4>::compress(
        m, (psz_buf<f8>*)m->buf, IN_d_data, OUT_d_compressed, OUT_compressed_bytes, m->stream);
    ((psz_buf<f8>*)m->buf)->reset(m->stream);
    if (status != PSZ_SUCCESS) return status;
    *OUT_header = *(m->header);
    memcpy_allkinds<H2D>((u1*)*OUT_d_compressed, (u1*)m->header, sizeof(psz_header));
    if (m->cli) CP<f8, u4>::compress_dump_internal_buf(m, (psz_buf<f8>*)m->buf, m->stream);
  }
  else {
    status = CP<f8, u2>::compress(
        m, (psz_buf<f8>*)m->buf, IN_d_data, OUT_d_compressed, OUT_compressed_bytes, m->stream);
    ((psz_buf<f8>*)m->buf)->reset(m->stream);
    if (status != PSZ_SUCCESS) return status;
    *OUT_header = *(m->header);
    memcpy_allkinds<H2D>((u1*)*OUT_d_compressed, (u1*)m->header, sizeof(psz_header));
    if (m->cli) CP<f8, u2>::compress_dump_internal_buf(m, (psz_buf<f8>*)m->buf, m->stream);
  }

  return status;
}

int psz_compress_analyze_float(psz_ctx* m, psz_rc2 rc, float* IN_d_data, u4* exported_h_hist)
{
  int status = PSZ_SUCCESS;

  RUNTIME_SAVE_CONFIG2();
  RUNTIME_CHANGE_EB_IF_REL(float);

  // TODO redundant
  m->header->eb = rc.eb;

  if (m->use_eq4)
    CP<f4, u4>::compress_analysis(
        m, (psz_buf<f4>*)m->buf, IN_d_data, exported_h_hist, m->stream);
  else
    CP<f4, u2>::compress_analysis(
        m, (psz_buf<f4>*)m->buf, IN_d_data, exported_h_hist, m->stream);

  return status;
}

int psz_decompress_float(
    psz_ctx* m, uint8_t* IN_d_compressed, size_t const IN_compressed_len,
    float* OUT_d_decompressed)
{
  bool const use_hfd_coarse = m->cli and m->cli->use_hfd_coarse;
  if (not pick_eq4(m)) return PSZ_ABORT_UNSUPPORTED_PIPELINE;
  return m->use_eq4 ? CP<f4, u4>::decompress(
                          m->header, (psz_buf<f4>*)m->buf, IN_d_compressed, OUT_d_decompressed,
                          m->stream, use_hfd_coarse)
                    : CP<f4, u2>::decompress(
                          m->header, (psz_buf<f4>*)m->buf, IN_d_compressed, OUT_d_decompressed,
                          m->stream, use_hfd_coarse);
}

int psz_decompress_double(
    psz_ctx* m, uint8_t* IN_d_compressed, size_t const IN_compressed_len,
    double* OUT_d_decompressed)
{
  bool const use_hfd_coarse = m->cli and m->cli->use_hfd_coarse;
  if (not pick_eq4(m)) return PSZ_ABORT_UNSUPPORTED_PIPELINE;
  return m->use_eq4 ? CP<f8, u4>::decompress(
                          m->header, (psz_buf<f8>*)m->buf, IN_d_compressed, OUT_d_decompressed,
                          m->stream, use_hfd_coarse)
                    : CP<f8, u2>::decompress(
                          m->header, (psz_buf<f8>*)m->buf, IN_d_compressed, OUT_d_decompressed,
                          m->stream, use_hfd_coarse);
}
