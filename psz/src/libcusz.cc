#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "compare.hh"
#include "compressor.hh"
#include "context_impl.h"
#include "cusz.h"
#include "mem/buf_comp.hh"
#include "module.hh"

using std::cerr;
using std::endl;
template <typename T>
using CP = psz::compressor_cpp<T>;

static thread_local psz_errno last_error = PSZ_SUCCESS;

psz_errno psz_last_error() { return last_error; }

const char* psz_error_string(int e)
{
  switch (e) {
    case PSZ_SUCCESS: return "success";
    // case PSZ_WARN_RADIUS_TOO_LARGE: return "radius too large";
    case PSZ_WARN_OUTLIER_TOO_MANY: return "more outliers than the outlier buffer holds";
    case PSZ_ABORT_UNSUPPORTED_TYPE: return "unsupported data type";
    case PSZ_ABORT_UNSUPPORTED_DIMENSION: return "unsupported dimensionality";
    case PSZ_ABORT_NOT_IMPLEMENTED: return "not implemented";
    // case PSZ_ABORT_NO_SUCH_PREDICTOR: return "no such predictor";
    case PSZ_ABORT_NO_SUCH_CODEC: return "no such codec";
    // case PSZ_ABORT_TOO_MANY_UNPREDICTABLE: return "too many unpredictables";
    // case PSZ_ABORT_TOO_MANY_ENC_BREAK: return "too many encoding breaks";
    case PSZ_ABORT_COMPRESSED_TOO_LARGE: return "compressed size exceeds the output buffer";
    case PSZ_ABORT_UNSUPPORTED_PIPELINE: return "unsupported pipeline";
    default: return "unknown error";
  }
}

static psz_ctx* fail(psz_errno status)
{
  last_error = status;
  return nullptr;
}

static psz_ctx* compress_ctx(psz_dtype dtype, psz_len len, void* stream)
{
  auto m = new psz_ctx;

  auto defaults = pszctx_default_values();
  m->header = new psz_header();
  memcpy(m->header, defaults->header, sizeof(psz_header));
  delete defaults;

  m->header->dtype = dtype;
  m->header->len = len;
  m->len_linear = len.x * len.y * len.z;
  m->bklen = m->header->radius * 2;
  m->cli = nullptr;
  m->stream = stream;

  last_error = PSZ_SUCCESS;
  return m;
}

psz_ctx* psz_compress_init(psz_dtype dtype, psz_len len, void* stream)
{
  auto m = compress_ctx(dtype, len, stream);
  if (dtype == F4)
    m->buf = CP<f4>::compress_init(m);
  else
    m->buf = CP<f8>::compress_init(m);
  return m;
}

psz_ctx* psz_compress_init_2stage(psz_dtype dtype, psz_len len, void* stream)
{ return psz_compress_init(dtype, len, stream); }

psz_ctx* psz_compress_init_3stage(psz_dtype dtype, psz_len len, void* stream)
{
  auto m = compress_ctx(dtype, len, stream);
  if (dtype == F4)
    m->buf = CP<f4>::compress_init_3stage(m);
  else
    m->buf = CP<f8>::compress_init_3stage(m);
  return m;
}

psz_ctx* psz_decompress_init(psz_header* header, void* stream)
{
  if (not psz::_2609::valid(header->pipeline)) return fail(PSZ_ABORT_UNSUPPORTED_PIPELINE);

  auto m = new psz_ctx;
  last_error = PSZ_SUCCESS;
  m->header = new psz_header();
  memcpy(m->header, header, sizeof(psz_header));
  m->bklen = m->header->radius * 2;
  m->len_linear = header->len.x * header->len.y * header->len.z;
  m->cli = nullptr;

  m->buf = header->dtype == F4 ? CP<f4>::decompress_init(m->header)
                               : CP<f8>::decompress_init(m->header);

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

#define RUNTIME_SAVE_CONFIG2()                                    \
  if (not psz::_2609::valid(m->header->pipeline) or not m->buf)   \
    return PSZ_ABORT_UNSUPPORTED_PIPELINE;                        \
  m->header->eb = eb;                                             \
  m->header->user_input_eb = eb;                                  \
  m->header->radius = psz::_2609::radius_of(m->header->pipeline); \
  m->bklen = m->header->radius * 2;

#define RUNTIME_SET_USER_INPUT_EB(Type)        \
  if (m->header->max_val > m->header->min_val) \
    m->header->user_input_eb = eb / ((Type)m->header->max_val - (Type)m->header->min_val);

psz_data_summary psz_compress_extrema_float(psz_ctx* m, float* IN_d_data)
{
  if (m->header->dtype != F4) {
    last_error = PSZ_ABORT_UNSUPPORTED_TYPE;
    return {NAN, NAN, NAN, NAN, NAN};
  }
  last_error = PSZ_SUCCESS;
  return CP<f4>::compress_extrema(m, IN_d_data, m->stream);
}

psz_data_summary psz_compress_extrema_double(psz_ctx* m, double* IN_d_data)
{
  if (m->header->dtype != F8) {
    last_error = PSZ_ABORT_UNSUPPORTED_TYPE;
    return {NAN, NAN, NAN, NAN, NAN};
  }
  last_error = PSZ_SUCCESS;
  return CP<f8>::compress_extrema(m, IN_d_data, m->stream);
}

int psz_compress_process_float(psz_ctx* m, psz_ppl pipeline, double eb, float* IN_d_data)
{
  if (m->header->dtype != F4) return PSZ_ABORT_UNSUPPORTED_TYPE;
  m->header->pipeline = pipeline;

  RUNTIME_SAVE_CONFIG2();
  RUNTIME_SET_USER_INPUT_EB(float);

  return CP<f4>::compress_process(m, pipeline, (psz_buf<f4>*)m->buf, IN_d_data, m->stream);
}

int psz_compress_process_double(psz_ctx* m, psz_ppl pipeline, double eb, double* IN_d_data)
{
  if (m->header->dtype != F8) return PSZ_ABORT_UNSUPPORTED_TYPE;
  m->header->pipeline = pipeline;

  RUNTIME_SAVE_CONFIG2();
  RUNTIME_SET_USER_INPUT_EB(double);

  return CP<f8>::compress_process(m, pipeline, (psz_buf<f8>*)m->buf, IN_d_data, m->stream);
}

int psz_compress_archive(
    psz_ctx* m, psz_header* OUT_header, uint8_t** OUT_d_compressed, size_t* OUT_compressed_bytes)
{
  if (not m->buf) return PSZ_ABORT_UNSUPPORTED_PIPELINE;
  auto const dtype = m->header->dtype;
  if (dtype == F4)
    return CP<f4>::compress_archive(
        m, (psz_buf<f4>*)m->buf, OUT_header, OUT_d_compressed, OUT_compressed_bytes, m->stream);
  else if (dtype == F8)
    return CP<f8>::compress_archive(
        m, (psz_buf<f8>*)m->buf, OUT_header, OUT_d_compressed, OUT_compressed_bytes, m->stream);
  else
    return PSZ_ABORT_UNSUPPORTED_TYPE;
}

int psz_compress_reset(psz_ctx* m)
{
  if (not m->buf) return PSZ_ABORT_UNSUPPORTED_PIPELINE;
  auto const dtype = m->header->dtype;
  if (dtype == F4)
    return CP<f4>::compress_reset(m, (psz_buf<f4>*)m->buf, m->stream);
  else if (dtype == F8)
    return CP<f8>::compress_reset(m, (psz_buf<f8>*)m->buf, m->stream);
  else
    return PSZ_ABORT_UNSUPPORTED_TYPE;
}

int psz_compress_analysis_float(psz_ctx* m, double eb, float* IN_d_data, u4* OUT_h_hist)
{
  if (m->header->dtype != F4) return PSZ_ABORT_UNSUPPORTED_TYPE;
  RUNTIME_SAVE_CONFIG2();
  RUNTIME_SET_USER_INPUT_EB(float);

  return CP<f4>::compress_analysis(m, (psz_buf<f4>*)m->buf, IN_d_data, OUT_h_hist, m->stream);
}

int psz_compress_analysis_double(psz_ctx* m, double eb, double* IN_d_data, u4* OUT_h_hist)
{
  if (m->header->dtype != F8) return PSZ_ABORT_UNSUPPORTED_TYPE;
  RUNTIME_SAVE_CONFIG2();
  RUNTIME_SET_USER_INPUT_EB(double);

  return CP<f8>::compress_analysis(m, (psz_buf<f8>*)m->buf, IN_d_data, OUT_h_hist, m->stream);
}

int psz_decompress_process_float(
    psz_ctx* m, uint8_t* IN_d_compressed, size_t const IN_compressed_len,
    float* OUT_d_decompressed)
{
  if (not m->buf) return PSZ_ABORT_UNSUPPORTED_PIPELINE;
  if (m->header->dtype != F4) return PSZ_ABORT_UNSUPPORTED_TYPE;
  return CP<f4>::decompress_process(
      m, (psz_buf<f4>*)m->buf, IN_d_compressed, OUT_d_decompressed, m->stream);
}

int psz_decompress_process_double(
    psz_ctx* m, uint8_t* IN_d_compressed, size_t const IN_compressed_len,
    double* OUT_d_decompressed)
{
  if (not m->buf) return PSZ_ABORT_UNSUPPORTED_PIPELINE;
  if (m->header->dtype != F8) return PSZ_ABORT_UNSUPPORTED_TYPE;
  return CP<f8>::decompress_process(
      m, (psz_buf<f8>*)m->buf, IN_d_compressed, OUT_d_decompressed, m->stream);
}

int psz_decompress_reset(psz_ctx* m)
{
  if (not m->buf) return PSZ_ABORT_UNSUPPORTED_PIPELINE;
  auto const dtype = m->header->dtype;
  if (dtype == F4)
    return CP<f4>::decompress_reset(m, (psz_buf<f4>*)m->buf, m->stream);
  else if (dtype == F8)
    return CP<f8>::decompress_reset(m, (psz_buf<f8>*)m->buf, m->stream);
  else
    return PSZ_ABORT_UNSUPPORTED_TYPE;
}
