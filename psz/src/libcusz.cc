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

psz_resource* psz_create_resource_manager(
    psz_dtype dtype, psz_len len, psz_pipeline pipeline, int spline_variant, void* stream)
{
  auto m = new psz_resource;

  auto defaults = pszctx_default_values();
  m->header = new psz_header;
  memcpy(m->header, defaults->header, sizeof(psz_header));
  delete defaults;

  m->header->dtype = dtype;
  m->header->pipeline = pipeline;
  m->header->len = len;
  m->len_linear = len.x * len.y * len.z;
  m->bklen = m->header->rc.radius * 2;
  m->spline_variant = spline_variant;  // creation-time; compress_init consumes it
  m->cli = nullptr;
  phf_coarse_tune(m->len_linear, &m->header->vle_sublen, &m->header->vle_pardeg);
  m->buf = dtype == F4 ? CP<f4, u2>::compress_init(m) : CP<f8, u2>::compress_init(m);
  m->stream = stream;

  return m;
}

psz_resource* psz_create_resource_manager_from_header(psz_header* header, void* stream)
{
  auto m = new psz_resource;
  m->header = new psz_header;
  memcpy(m->header, header, sizeof(psz_header));
  m->bklen = m->header->rc.radius * 2;
  m->len_linear = header->len.x * header->len.y * header->len.z;
  m->spline_variant = 0;  // default y25; variant is not (yet) serialized in the header
  m->cli = nullptr;

  m->buf = header->dtype == F4 ? CP<f4, u2>::decompress_init(m->header)
                               : CP<f8, u2>::decompress_init(m->header);

  m->stream = stream;

  return m;
}

void psz_modify_resource_manager_from_header(psz_resource* manager, psz_header* header)
{
  memcpy(manager->header, header, sizeof(psz_header));
  manager->bklen = manager->header->rc.radius * 2;
  manager->len_linear = header->len.x * header->len.y * header->len.z;
}

int psz_release_resource(psz_resource* manager)
{
  auto dtype = manager->header->dtype;
  if (dtype == F4) {
    if (manager->buf) delete (psz::Buf_Comp<f4, u2>*)manager->buf;
  }
  else if (dtype == F8) {
    if (manager->buf) delete (psz::Buf_Comp<f8, u2>*)manager->buf;
  }
  else
    return PSZ_ABORT_UNSUPPORTED_TYPE;

  if (manager->cli) delete manager->cli;
  if (manager->header) delete manager->header;
  delete manager;

  return 0;
}

#define RUNTIME_SAVE_CONFIG2()      \
  m->header->rc = rc;               \
  m->header->user_input_eb = rc.eb; \
  m->bklen = rc.radius * 2;

#define RUNTIME_CHECK_RADIUS(Type)                   \
  if (rc.radius > psz::Buf_Comp<Type>::max_radius) { \
    rc.radius = psz::Buf_Comp<Type>::max_radius;     \
    status = PSZ_WARN_RADIUS_TOO_LARGE;              \
  }

#define RUNTIME_CHANGE_EB_IF_REL(Type)                                                         \
  if (rc.mode == Rel) {                                                                        \
    auto [min_val, max_val, avg_val, rng] = psz::cuda::GPU_get_extrema<Type>::kernel(IN_d_data, m->len_linear); \
    (void)avg_val;                                                                             \
    m->header->min_val = min_val;                                                              \
    m->header->max_val = max_val;                                                              \
    m->header->rc.eb *= rng;                                                                   \
  }

int psz_compress_float(
    psz_resource* m, psz_rc2 rc, float* IN_d_data, psz_header* OUT_header,
    uint8_t** OUT_d_compressed, size_t* OUT_compressed_bytes)
{
  int status = PSZ_SUCCESS;

  RUNTIME_CHECK_RADIUS(float);
  RUNTIME_SAVE_CONFIG2();
  RUNTIME_CHANGE_EB_IF_REL(float);

  CP<f4, u2>::compress(
      m, (psz_buf<f4, u2>*)m->buf, IN_d_data, OUT_d_compressed, OUT_compressed_bytes, m->stream);
  *OUT_header = *(m->header);
  if (m->cli) CP<f4, u2>::compress_dump_internal_buf(m, (psz_buf<f4, u2>*)m->buf, m->stream);

  return status;
}

int psz_compress_double(
    psz_resource* m, psz_rc2 rc, double* IN_d_data, psz_header* OUT_header,
    uint8_t** OUT_d_compressed, size_t* OUT_compressed_bytes)
{
  int status = PSZ_SUCCESS;

  RUNTIME_CHECK_RADIUS(double);
  RUNTIME_SAVE_CONFIG2();
  RUNTIME_CHANGE_EB_IF_REL(double);

  CP<f8, u2>::compress(
      m, (psz_buf<f8, u2>*)m->buf, IN_d_data, OUT_d_compressed, OUT_compressed_bytes, m->stream);
  *OUT_header = *(m->header);
  if (m->cli) CP<f8, u2>::compress_dump_internal_buf(m, (psz_buf<f8, u2>*)m->buf, m->stream);

  return status;
}

int psz_compress_analyize_float(psz_resource* m, psz_rc2 rc, float* IN_d_data, u4* exported_h_hist)
{
  int status = PSZ_SUCCESS;

  RUNTIME_CHECK_RADIUS(float);
  RUNTIME_SAVE_CONFIG2();
  RUNTIME_CHANGE_EB_IF_REL(float);

  // TODO redundant
  m->header->rc.eb = rc.eb;

  CP<f4, u2>::compress_analysis(
      m, (psz_buf<f4, u2>*)m->buf, IN_d_data, exported_h_hist, m->stream);

  return status;
}

int psz_decompress_float(
    psz_resource* m, uint8_t* IN_d_compressed, size_t const IN_compressed_len,
    float* OUT_d_decompressed)
{
  CP<f4, u2>::decompress(
      m->header, (psz_buf<f4, u2>*)m->buf, IN_d_compressed, OUT_d_decompressed, m->stream);

  return PSZ_SUCCESS;
}

int psz_decompress_double(
    psz_resource* m, uint8_t* IN_d_compressed, size_t const IN_compressed_len,
    double* OUT_d_decompressed)
{
  CP<f8, u2>::decompress(
      m->header, (psz_buf<f8, u2>*)m->buf, IN_d_compressed, OUT_d_decompressed, m->stream);

  return PSZ_SUCCESS;
}
