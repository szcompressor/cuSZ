/**
 * @file cusz_lib.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-05-01
 * (rev.1) 2023-01-29
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "compressor.hh"
#include "cusz.h"
#include "cusz/context.h"
#include "cusz/type.h"
#include "cusz_rev1.h"
#include "detail/compare.hh"
#include "mem/buf_comp.hh"

using psz::analysis::CPU_probe_extrema;
using psz::analysis::GPU_probe_extrema;

// using _portable::memobj;
using TimeRecordTuple = std::tuple<const char*, double>;
using TimeRecord = std::vector<TimeRecordTuple>;
using timerecord_t = TimeRecord*;

psz_compressor* psz_create(
    psz_dtype const dtype, psz_len3 const uncomp_len3, psz_predictor const predictor,
    int const quantizer_radius, psz_codec const codec)
{
  auto comp = new psz_compressor{
      .compressor = nullptr,
      .ctx = pszctx_minimal_workset(dtype, predictor, quantizer_radius, codec),
      .last_error = CUSZ_SUCCESS};

  comp->ctx->header->dtype = dtype;

  pszctx_set_len(comp->ctx, uncomp_len3);
  phf_coarse_tune(
      comp->ctx->len_linear, &comp->ctx->header->vle_sublen, &comp->ctx->header->vle_pardeg);
  if (dtype == F4 or dtype == F8)
    comp->compressor = dtype == F4 ? (void*)(new psz::CompressorF4(comp->ctx))
                                   : (void*)(new psz::CompressorF8(comp->ctx));
  else
    comp->last_error = PSZ_ABORT_UNSUPPORTED_TYPE;

  return comp;
}

psz_compressor* psz_create_default(psz_dtype const dtype, psz_len3 const uncomp_len3)
{
  auto comp = new psz_compressor{
      .compressor = nullptr, .ctx = pszctx_default_values(), .last_error = CUSZ_SUCCESS};

  comp->ctx->header->dtype = dtype;

  pszctx_set_len(comp->ctx, uncomp_len3);
  phf_coarse_tune(
      comp->ctx->len_linear, &comp->ctx->header->vle_sublen, &comp->ctx->header->vle_pardeg);
  if (dtype == F4 or dtype == F8)
    comp->compressor = dtype == F4 ? (void*)(new psz::CompressorF4(comp->ctx))
                                   : (void*)(new psz::CompressorF8(comp->ctx));
  else
    comp->last_error = PSZ_ABORT_UNSUPPORTED_TYPE;

  return comp;
}

psz_compressor* psz_create_from_context(psz_ctx* const ctx, psz_len3 uncomp_len3)
{
  auto comp = new psz_compressor{.compressor = nullptr, .ctx = ctx, .last_error = CUSZ_SUCCESS};

  pszctx_set_len(comp->ctx, uncomp_len3);
  auto dtype = ctx->header->dtype;

  phf_coarse_tune(
      comp->ctx->len_linear, &comp->ctx->header->vle_sublen, &comp->ctx->header->vle_pardeg);
  if (dtype == F4 or dtype == F8)
    comp->compressor = dtype == F4 ? (void*)(new psz::CompressorF4(comp->ctx))
                                   : (void*)(new psz::CompressorF8(comp->ctx));
  else
    comp->last_error = PSZ_ABORT_UNSUPPORTED_TYPE;

  return comp;
}

psz_compressor* psz_create_from_header(psz_header* const h)
{
  auto comp = new psz_compressor{
      .compressor = nullptr, .ctx = pszctx_default_values(), .last_error = CUSZ_SUCCESS};

  comp->ctx->header = h;

  if (h->dtype == F4 or h->dtype == F8)
    comp->compressor = h->dtype == F4 ? (void*)(new psz::CompressorF4(comp->ctx))
                                      : (void*)(new psz::CompressorF8(comp->ctx));
  else
    comp->last_error = PSZ_ABORT_UNSUPPORTED_TYPE;

  return comp;
}

pszerror psz_release(psz_compressor* comp)
{
  auto dtype = comp->ctx->header->dtype;
  if (dtype == F4)
    delete (psz::CompressorF4*)comp->compressor;
  else if (dtype == F8)
    delete (psz::CompressorF8*)comp->compressor;
  else
    return PSZ_ABORT_UNSUPPORTED_TYPE;

  delete comp;
  return CUSZ_SUCCESS;
}

pszerror psz_compress(
    psz_compressor* comp, void* d_in, psz_len3 const in_len3, double const eb, psz_mode const mode,
    uint8_t** comped, size_t* comp_bytes, psz_header* header, void* record, void* stream)
{
  comp->ctx->header->rc.eb = eb;
  comp->ctx->header->rc.mode = mode;
  comp->ctx->header->user_input_eb = eb;
  auto len = in_len3.x * in_len3.y * in_len3.z;
  auto dtype = comp->ctx->header->dtype;

  if (mode == Rel) {
    double _max_val, _min_val, _rng;
    if (dtype == F4)
      GPU_probe_extrema((f4*)d_in, len, _max_val, _min_val, _rng);
    else
      GPU_probe_extrema((f8*)d_in, len, _max_val, _min_val, _rng);

    comp->ctx->header->rc.eb *= _rng;
    comp->ctx->header->max_val = _max_val;
    comp->ctx->header->min_val = _min_val;
  }

  if (dtype == F4) {
    auto cor = (psz::CompressorF4*)(comp->compressor);

    cor->compress(comp->ctx, (f4*)(d_in), comped, comp_bytes, stream);
    cor->export_header(*header);
    // cor->export_timerecord((psz::TimeRecord*)record);
    // cor->export_timerecord(comp->stage_time);
    cor->dump_compress_intermediate(comp->ctx, stream);
  }
  else if (dtype == F8) {
    auto cor = (psz::CompressorF8*)(comp->compressor);

    cor->compress(comp->ctx, (f8*)(d_in), comped, comp_bytes, stream);
    cor->export_header(*header);
    // cor->export_timerecord((psz::TimeRecord*)record);
    // cor->export_timerecord(comp->stage_time);
    cor->dump_compress_intermediate(comp->ctx, stream);
  }
  else {
    // TODO put to log-queue
    cerr << std::string(__FUNCTION__) + ": Type is not supported." << endl;
    return PSZ_ABORT_UNSUPPORTED_TYPE;
  }

  return CUSZ_SUCCESS;
}

pszerror psz_decompress(
    psz_compressor* comp, uint8_t* comped, size_t const comp_len, void* decomped,
    psz_len3 const decomp_len, void* record, void* stream)
{
  auto dtype = comp->ctx->header->dtype;
  if (dtype == F4) {
    auto cor = (psz::CompressorF4*)(comp->compressor);

    cor->decompress(comp->ctx->header, comped, (f4*)(decomped), (cudaStream_t)stream);
    // cor->export_timerecord((psz::TimeRecord*)record);
    // cor->export_timerecord(comp->stage_time);
  }
  else if (dtype == F8) {
    auto cor = (psz::CompressorF8*)(comp->compressor);

    cor->decompress(comp->ctx->header, comped, (f8*)(decomped), (cudaStream_t)stream);
    // cor->export_timerecord((psz::TimeRecord*)record);
    // cor->export_timerecord(comp->stage_time);
  }
  else {
    // TODO put to log-queue
    cerr << std::string(__FUNCTION__) + ": Type is not supported." << endl;
    return PSZ_ABORT_UNSUPPORTED_TYPE;
  }

  return CUSZ_SUCCESS;
}

pszerror psz_clear_buffer(psz_compressor* comp)
{
  auto dtype = comp->ctx->header->dtype;
  if (dtype == F4) {
    auto cor = (psz::CompressorF4*)(comp->compressor);
    cor->clear_buffer();
  }
  else if (dtype == F8) {
    auto cor = (psz::CompressorF8*)(comp->compressor);
    cor->clear_buffer();
  }
  else {
    // TODO put to log-queue
    cerr << std::string(__FUNCTION__) + ": Type is not supported." << endl;
    return PSZ_ABORT_UNSUPPORTED_TYPE;
  }

  return CUSZ_SUCCESS;
}

template <typename T, typename E>
using CP = psz::compression_pipeline<T, E>;

psz_resource* psz_create_resource_manager(
    psz_dtype dtype, psz_len len, psz_pipeline pipeline, void* stream)
{
  auto m = new psz_resource;

  m->header = new psz_header{.dtype = dtype, .pipeline = pipeline, .len = len};
  m->len_linear = len.x * len.y * len.z;
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
  m->dict_size = m->header->rc.radius * 2;
  m->len_linear = header->len.x * header->len.y * header->len.z;
  m->cli = nullptr;

  m->buf = header->dtype == F4 ? CP<f4, u2>::compress_init(m) : CP<f8, u2>::compress_init(m);

  m->stream = stream;

  return m;
}

void psz_modify_resource_manager_from_header(psz_resource* manager, psz_header* header)
{
  memcpy(manager->header, header, sizeof(psz_header));
  manager->dict_size = manager->header->rc.radius * 2;
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
  m->dict_size = rc.radius * 2;

#define RUNTIME_CHECK_RADIUS(Type)                   \
  if (rc.radius > psz::Buf_Comp<Type>::max_radius) { \
    rc.radius = psz::Buf_Comp<Type>::max_radius;     \
    status = PSZ_WARN_RADIUS_TOO_LARGE;              \
  }

#define RUNTIME_CHANGE_EB_IF_REL(Type)                                           \
  if (rc.mode == Rel) {                                                          \
    double _max_val, _min_val, _rng;                                             \
    GPU_probe_extrema<Type>(                                                     \
        IN_d_data, m->len_linear, m->header->max_val, m->header->min_val, _rng); \
    m->header->rc.eb *= _rng;                                                    \
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
  // if (m->cli) CP<f4, u2>::compress_dump_internal_buf(m, (psz_buf<f4, u2>*)m->buf, m->stream);

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