#include "context_impl.h"
#include "pipeline.h"
#include <iostream>

#include "compare.hh"
#include "compressor.hh"
#include "cusz.h"
#include "extrema.hh"
#include "mem/buf_comp.hh"

using std::cerr;
using std::endl;
template <typename T, typename E>
using CP = psz::compression_pipeline<T, E>;

// why the last creator returned NULL; per-thread, so concurrent creation on
// separate streams does not overwrite one another's reason
static thread_local psz_error_status last_error = PSZ_SUCCESS;

psz_error_status psz_last_error() { return last_error; }

static psz_ctx* fail(psz_error_status status)
{
  last_error = status;
  return nullptr;
}

// eq/SYM width follows the pipeline: the HFR encoders quantize into u4, everything
// else into u2, so a caller never has to know which width its pipeline implies.
static psz_ctx* make_manager(
    psz_dtype dtype, psz_len len, psz_ppl pipeline, void* stream)
{
  if (not pszppl_supported(pipeline)) return fail(PSZ_ABORT_UNSUPPORTED_PIPELINE);

  auto m = new psz_ctx;

  auto defaults = pszctx_default_values();
  m->header = new psz_header();
  memcpy(m->header, defaults->header, sizeof(psz_header));
  delete defaults;

  m->header->dtype = dtype;
  m->header->pipeline = pipeline;
  m->header->len = len;
  m->len_linear = len.x * len.y * len.z;
  m->header->radius = pszppl_radius(pipeline);  // HFR books assume 128
  m->bklen = m->header->radius * 2;
  m->cli = nullptr;
  m->use_eq4 = pszppl_needs_eq4(pipeline) != 0;
  m->buf = m->use_eq4
               ? (dtype == F4 ? CP<f4, u4>::compress_init(m) : CP<f8, u4>::compress_init(m))
               : (dtype == F4 ? CP<f4, u2>::compress_init(m) : CP<f8, u2>::compress_init(m));
  m->stream = stream;

  last_error = PSZ_SUCCESS;
  return m;
}

psz_ctx* psz_init(
    psz_dtype dtype, psz_len len, psz_ppl pipeline, void* stream)
{
  return make_manager(dtype, len, pipeline, stream);
}

// Stages rather than a filled-in psz_ppl: compose derives hist from codec1
// and resolves a pass-2 request into the chain that fits, so the caller names
// only what it actually chooses.
psz_ctx* psz_init_from_stages(
    psz_dtype dtype, psz_len len, psz_predictor p1, psz_codec c1, psz_codec optional_c2,
    void* stream)
{
  return make_manager(dtype, len, pszppl_compose(p1, c1, optional_c2), stream);
}

// A preset fixes what psz_init takes piecemeal: the pipeline.
psz_ctx* psz_init_from_preset(
    psz_dtype dtype, psz_len len, psz_preset preset, void* stream)
{
  // a generic preset names a shape, not a pipeline
  if (pszpreset_is_generic(preset)) return fail(PSZ_ABORT_UNSUPPORTED_PIPELINE);

  return make_manager(dtype, len, pszpreset_pipeline(preset), stream);
}

psz_ctx* psz_init_from_header(psz_header* header, void* stream)
{
  if (not pszppl_supported(header->pipeline)) return fail(PSZ_ABORT_UNSUPPORTED_PIPELINE);

  auto m = new psz_ctx;
  last_error = PSZ_SUCCESS;
  m->header = new psz_header();
  memcpy(m->header, header, sizeof(psz_header));
  m->bklen = m->header->radius * 2;
  m->len_linear = header->len.x * header->len.y * header->len.z;
  m->cli = nullptr;
  m->use_eq4 = pszppl_needs_eq4(header->pipeline) != 0;

  m->buf = m->use_eq4 ? (header->dtype == F4 ? CP<f4, u4>::decompress_init(m->header)
                                             : CP<f8, u4>::decompress_init(m->header))
                      : (header->dtype == F4 ? CP<f4, u2>::decompress_init(m->header)
                                             : CP<f8, u2>::decompress_init(m->header));

  m->stream = stream;

  return m;
}

int psz_free(psz_ctx* manager)
{
  auto dtype = manager->header->dtype;
  auto eq4 = manager->use_eq4;
  if (dtype == F4) {
    if (manager->buf)
      eq4 ? delete (psz::Buf_Comp<f4, u4>*)manager->buf
          : delete (psz::Buf_Comp<f4, u2>*)manager->buf;
  }
  else if (dtype == F8) {
    if (manager->buf)
      eq4 ? delete (psz::Buf_Comp<f8, u4>*)manager->buf
          : delete (psz::Buf_Comp<f8, u2>*)manager->buf;
  }
  else
    return PSZ_ABORT_UNSUPPORTED_TYPE;

  if (manager->cli) delete manager->cli;
  if (manager->header) delete manager->header;
  delete manager;

  return 0;
}

#define RUNTIME_SAVE_CONFIG2()      \
  m->header->eb = rc.eb;         \
  m->header->user_input_eb = rc.eb; \
  m->bklen = m->header->radius * 2;

// radius 0 means the radius the manager was created with
#define RUNTIME_CHANGE_EB_IF_REL(Type)                                      \
  if (rc.mode == Rel) {                                                     \
    auto [min_val, max_val, avg_val, rng] =                                 \
        psz::cuda::GPU_get_extrema<Type>::kernel(IN_d_data, m->len_linear); \
    (void)avg_val;                                                          \
    m->header->min_val = min_val;                                           \
    m->header->max_val = max_val;                                           \
    m->header->eb *= rng;                                                \
  }

int psz_compress_float(
    psz_ctx* m, psz_rc2 rc, float* IN_d_data, psz_header* OUT_header,
    uint8_t** OUT_d_compressed, size_t* OUT_compressed_bytes)
{
  int status = PSZ_SUCCESS;

  RUNTIME_SAVE_CONFIG2();
  RUNTIME_CHANGE_EB_IF_REL(float);

  if (m->use_eq4) {
    status = CP<f4, u4>::compress(
        m, (psz_buf<f4, u4>*)m->buf, IN_d_data, OUT_d_compressed, OUT_compressed_bytes, m->stream);
    if (status != PSZ_SUCCESS) return status;
    *OUT_header = *(m->header);
    memcpy_allkinds<H2D>((u1*)*OUT_d_compressed, (u1*)m->header, sizeof(psz_header));
    if (m->cli) CP<f4, u4>::compress_dump_internal_buf(m, (psz_buf<f4, u4>*)m->buf, m->stream);
  }
  else {
    status = CP<f4, u2>::compress(
        m, (psz_buf<f4, u2>*)m->buf, IN_d_data, OUT_d_compressed, OUT_compressed_bytes, m->stream);
    ((psz_buf<f4, u2>*)m->buf)->reset(m->stream);
    if (status != PSZ_SUCCESS) return status;
    *OUT_header = *(m->header);
    memcpy_allkinds<H2D>((u1*)*OUT_d_compressed, (u1*)m->header, sizeof(psz_header));
    if (m->cli) CP<f4, u2>::compress_dump_internal_buf(m, (psz_buf<f4, u2>*)m->buf, m->stream);
  }

  return status;
}

int psz_compress_double(
    psz_ctx* m, psz_rc2 rc, double* IN_d_data, psz_header* OUT_header,
    uint8_t** OUT_d_compressed, size_t* OUT_compressed_bytes)
{
  int status = PSZ_SUCCESS;

  RUNTIME_SAVE_CONFIG2();
  RUNTIME_CHANGE_EB_IF_REL(double);

  if (m->use_eq4) {
    status = CP<f8, u4>::compress(
        m, (psz_buf<f8, u4>*)m->buf, IN_d_data, OUT_d_compressed, OUT_compressed_bytes, m->stream);
    if (status != PSZ_SUCCESS) return status;
    *OUT_header = *(m->header);
    memcpy_allkinds<H2D>((u1*)*OUT_d_compressed, (u1*)m->header, sizeof(psz_header));
    if (m->cli) CP<f8, u4>::compress_dump_internal_buf(m, (psz_buf<f8, u4>*)m->buf, m->stream);
  }
  else {
    status = CP<f8, u2>::compress(
        m, (psz_buf<f8, u2>*)m->buf, IN_d_data, OUT_d_compressed, OUT_compressed_bytes, m->stream);
    ((psz_buf<f8, u2>*)m->buf)->reset(m->stream);
    if (status != PSZ_SUCCESS) return status;
    *OUT_header = *(m->header);
    memcpy_allkinds<H2D>((u1*)*OUT_d_compressed, (u1*)m->header, sizeof(psz_header));
    if (m->cli) CP<f8, u2>::compress_dump_internal_buf(m, (psz_buf<f8, u2>*)m->buf, m->stream);
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
        m, (psz_buf<f4, u4>*)m->buf, IN_d_data, exported_h_hist, m->stream);
  else
    CP<f4, u2>::compress_analysis(
        m, (psz_buf<f4, u2>*)m->buf, IN_d_data, exported_h_hist, m->stream);

  return status;
}

int psz_decompress_float(
    psz_ctx* m, uint8_t* IN_d_compressed, size_t const IN_compressed_len,
    float* OUT_d_decompressed)
{
  bool const use_hfd_coarse = m->cli and m->cli->use_hfd_coarse;
  return m->use_eq4 ? CP<f4, u4>::decompress(
                          m->header, (psz_buf<f4, u4>*)m->buf, IN_d_compressed, OUT_d_decompressed,
                          m->stream, use_hfd_coarse)
                    : CP<f4, u2>::decompress(
                          m->header, (psz_buf<f4, u2>*)m->buf, IN_d_compressed, OUT_d_decompressed,
                          m->stream, use_hfd_coarse);
}

int psz_decompress_double(
    psz_ctx* m, uint8_t* IN_d_compressed, size_t const IN_compressed_len,
    double* OUT_d_decompressed)
{
  bool const use_hfd_coarse = m->cli and m->cli->use_hfd_coarse;
  return m->use_eq4 ? CP<f8, u4>::decompress(
                          m->header, (psz_buf<f8, u4>*)m->buf, IN_d_compressed, OUT_d_decompressed,
                          m->stream, use_hfd_coarse)
                    : CP<f8, u2>::decompress(
                          m->header, (psz_buf<f8, u2>*)m->buf, IN_d_compressed, OUT_d_decompressed,
                          m->stream, use_hfd_coarse);
}
