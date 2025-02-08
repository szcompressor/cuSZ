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

#include "api_v2.h"
#include "compressor.hh"
#include "context.h"
#include "cusz.h"
#include "cusz/type.h"
#include "stat/compare.hh"

using psz::analysis::CPU_probe_extrema;
using psz::analysis::GPU_probe_extrema;

// using _portable::memobj;
using TimeRecordTuple = std::tuple<const char*, double>;
using TimeRecord = std::vector<TimeRecordTuple>;
using timerecord_t = TimeRecord*;

psz_compressor* capi_psz_create(
    psz_dtype const dtype, psz_len3 const uncomp_len3, psz_predtype const predictor,
    int const quantizer_radius, psz_codectype const codec)
{
  auto comp = new psz_compressor{
      .compressor = nullptr,
      .ctx = pszctx_minimal_workset(dtype, predictor, quantizer_radius, codec),
      .last_error = CUSZ_SUCCESS};

  comp->ctx->header->dtype = dtype;

  pszctx_set_len(comp->ctx, uncomp_len3);
  capi_phf_coarse_tune(
      comp->ctx->data_len, &comp->ctx->header->vle_sublen, &comp->ctx->header->vle_pardeg);
  if (dtype == F4 or dtype == F8)
    comp->compressor = dtype == F4 ? (void*)(new psz::CompressorF4(comp->ctx))
                                   : (void*)(new psz::CompressorF8(comp->ctx));
  else
    comp->last_error = PSZ_TYPE_UNSUPPORTED;

  return comp;
}

psz_compressor* capi_psz_create_default(psz_dtype const dtype, psz_len3 const uncomp_len3)
{
  auto comp = new psz_compressor{
      .compressor = nullptr, .ctx = pszctx_default_values(), .last_error = CUSZ_SUCCESS};

  comp->ctx->header->dtype = dtype;

  pszctx_set_len(comp->ctx, uncomp_len3);
  capi_phf_coarse_tune(
      comp->ctx->data_len, &comp->ctx->header->vle_sublen, &comp->ctx->header->vle_pardeg);
  if (dtype == F4 or dtype == F8)
    comp->compressor = dtype == F4 ? (void*)(new psz::CompressorF4(comp->ctx))
                                   : (void*)(new psz::CompressorF8(comp->ctx));
  else
    comp->last_error = PSZ_TYPE_UNSUPPORTED;

  return comp;
}

psz_compressor* capi_psz_create_from_context(pszctx* const ctx, psz_len3 uncomp_len3)
{
  auto comp = new psz_compressor{.compressor = nullptr, .ctx = ctx, .last_error = CUSZ_SUCCESS};

  pszctx_set_len(comp->ctx, uncomp_len3);
  auto dtype = ctx->header->dtype;

  capi_phf_coarse_tune(
      comp->ctx->data_len, &comp->ctx->header->vle_sublen, &comp->ctx->header->vle_pardeg);
  if (dtype == F4 or dtype == F8)
    comp->compressor = dtype == F4 ? (void*)(new psz::CompressorF4(comp->ctx))
                                   : (void*)(new psz::CompressorF8(comp->ctx));
  else
    comp->last_error = PSZ_TYPE_UNSUPPORTED;

  return comp;
}

psz_compressor* capi_psz_create_from_header(psz_header* const h)
{
  auto comp = new psz_compressor{
      .compressor = nullptr, .ctx = pszctx_default_values(), .last_error = CUSZ_SUCCESS};

  comp->ctx->header = h;

  if (h->dtype == F4 or h->dtype == F8)
    comp->compressor = h->dtype == F4 ? (void*)(new psz::CompressorF4(comp->ctx))
                                      : (void*)(new psz::CompressorF8(comp->ctx));
  else
    comp->last_error = PSZ_TYPE_UNSUPPORTED;

  return comp;
}

pszerror capi_psz_release(psz_compressor* comp)
{
  auto dtype = comp->ctx->header->dtype;
  if (dtype == F4)
    delete (psz::CompressorF4*)comp->compressor;
  else if (dtype == F8)
    delete (psz::CompressorF8*)comp->compressor;
  else
    return PSZ_TYPE_UNSUPPORTED;

  delete comp;
  return CUSZ_SUCCESS;
}

pszerror capi_psz_compress(
    psz_compressor* comp, void* d_in, psz_len3 const in_len3, double const eb, psz_mode const mode,
    uint8_t** comped, size_t* comp_bytes, psz_header* header, void* record, void* stream)
{
  comp->ctx->header->eb = eb;
  comp->ctx->header->mode = mode;
  comp->ctx->header->user_input_eb = eb;
  auto len = in_len3.x * in_len3.y * in_len3.z;
  auto dtype = comp->ctx->header->dtype;

  if (mode == Rel) {
    double _max_val, _min_val, _rng;
    if (dtype == F4)
      GPU_probe_extrema((f4*)d_in, len, _max_val, _min_val, _rng);
    else
      GPU_probe_extrema((f8*)d_in, len, _max_val, _min_val, _rng);

    comp->ctx->header->eb *= _rng;
    comp->ctx->header->logging_max = _max_val;
    comp->ctx->header->logging_min = _min_val;
  }

  if (dtype == F4) {
    auto cor = (psz::CompressorF4*)(comp->compressor);

    cor->compress(comp->ctx, (f4*)(d_in), comped, comp_bytes, stream);
    cor->export_header(*header);
    cor->export_timerecord((psz::TimeRecord*)record);
    cor->export_timerecord(comp->stage_time);
    cor->dump_compress_intermediate(comp->ctx, stream);
  }
  else if (dtype == F8) {
    auto cor = (psz::CompressorF8*)(comp->compressor);

    cor->compress(comp->ctx, (f8*)(d_in), comped, comp_bytes, stream);
    cor->export_header(*header);
    cor->export_timerecord((psz::TimeRecord*)record);
    cor->export_timerecord(comp->stage_time);
    cor->dump_compress_intermediate(comp->ctx, stream);
  }
  else {
    // TODO put to log-queue
    cerr << std::string(__FUNCTION__) + ": Type is not supported." << endl;
    return PSZ_TYPE_UNSUPPORTED;
  }

  return CUSZ_SUCCESS;
}

pszerror capi_psz_decompress(
    psz_compressor* comp, uint8_t* comped, size_t const comp_len, void* decomped,
    psz_len3 const decomp_len, void* record, void* stream)
{
  auto dtype = comp->ctx->header->dtype;
  if (dtype == F4) {
    auto cor = (psz::CompressorF4*)(comp->compressor);

    cor->decompress(comp->ctx->header, comped, (f4*)(decomped), (cudaStream_t)stream);
    cor->export_timerecord((psz::TimeRecord*)record);
    cor->export_timerecord(comp->stage_time);
  }
  else if (dtype == F8) {
    auto cor = (psz::CompressorF8*)(comp->compressor);

    cor->decompress(comp->ctx->header, comped, (f8*)(decomped), (cudaStream_t)stream);
    cor->export_timerecord((psz::TimeRecord*)record);
    cor->export_timerecord(comp->stage_time);
  }
  else {
    // TODO put to log-queue
    cerr << std::string(__FUNCTION__) + ": Type is not supported." << endl;
    return PSZ_TYPE_UNSUPPORTED;
  }

  return CUSZ_SUCCESS;
}

pszerror capi_psz_clear_buffer(psz_compressor* comp)
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
    return PSZ_TYPE_UNSUPPORTED;
  }

  return CUSZ_SUCCESS;
}

psz_resource* psz_create_resource_manager(
    psz_dtype t, uint32_t x, uint32_t y, uint32_t z, void* stream)
{
  auto m = new psz_resource;

  // FUTURE: .radius -> .max_radius; adjust radius in RC
  m->header = new psz_header{
      .dtype = t,
      .pred_type = Lorenzo,
      .hist_type = HistogramGeneric,
      .codec1_type = Huffman,
      .radius = 512,
      .x = x,
      .y = y,
      .z = z,
      .w = 1,
  };

  m->dict_size = m->header->radius * 2;
  m->data_len = x * y * z;
  // m->ndim = 3;
  m->cli = nullptr;

  capi_phf_coarse_tune(m->data_len, &m->header->vle_sublen, &m->header->vle_pardeg);

  m->compressor = t == F4 ? (void*)(new psz::CompressorF4(m)) : (void*)(new psz::CompressorF8(m));
  m->stream = stream;

  return m;
}

psz_resource* psz_create_resource_manager_from_header(psz_header* header, void* stream)
{
  auto m = new psz_resource;
  m->header = new psz_header;
  memcpy(m->header, header, sizeof(psz_header));
  m->dict_size = m->header->radius * 2;
  m->data_len = header->x * header->y * header->z;
  m->cli = nullptr;
  m->compressor =
      header->dtype == F4 ? (void*)(new psz::CompressorF4(m)) : (void*)(new psz::CompressorF8(m));
  m->stream = stream;

  return m;
}

void psz_modify_resource_manager_from_header(psz_resource* manager, psz_header* header)
{
  memcpy(manager->header, header, sizeof(psz_header));
  manager->dict_size = manager->header->radius * 2;
  manager->data_len = header->x * header->y * header->z;
}

int psz_release_resource(psz_resource* manager)
{
  auto dtype = manager->header->dtype;
  if (dtype == F4)
    delete (psz::CompressorF4*)manager->compressor;
  else if (dtype == F8)
    delete (psz::CompressorF8*)manager->compressor;
  else
    return PSZ_TYPE_UNSUPPORTED;

  if (manager->cli) delete manager->cli;
  if (manager->header) delete manager->header;
  delete manager;

  return 0;
}

int psz_compress_float(
    psz_resource* m, psz_rc rc, float* IN_d_data, psz_header* OUT_compressed_metadata,
    uint8_t** OUT_dptr_compressed, size_t* OUT_compressed_bytes)
{
  m->header->pred_type = rc.predictor;
  m->header->codec1_type = rc.codec1;
  m->header->hist_type = rc.hist;
  m->header->mode = rc.mode;
  m->header->eb = rc.eb;
  m->header->user_input_eb = rc.eb;

  // TODO check type
  auto dtype = m->header->dtype;

  if (rc.mode == Rel) {
    double _max_val, _min_val, _rng;
    GPU_probe_extrema<float>(IN_d_data, m->data_len, _max_val, _min_val, _rng);
    m->header->eb *= _rng;
    m->header->logging_max = _max_val;
    m->header->logging_min = _min_val;
  }

  auto c = (psz::CompressorF4*)m->compressor;
  c->compress(m, IN_d_data, OUT_dptr_compressed, OUT_compressed_bytes, m->stream);
  c->export_header(*OUT_compressed_metadata);

  return 0;
}

int psz_compress_double(
    psz_resource* m, psz_rc rc, double* IN_d_data, psz_header* OUT_compressed_metadata,
    uint8_t** OUT_dptr_compressed, size_t* OUT_compressed_bytes)
{
  m->header->pred_type = rc.predictor;
  m->header->mode = rc.mode;
  m->header->eb = rc.eb;
  m->header->user_input_eb = rc.eb;

  // TODO check type
  auto dtype = m->header->dtype;

  if (rc.mode == Rel) {
    double _max_val, _min_val, _rng;
    GPU_probe_extrema<double>(IN_d_data, m->data_len, _max_val, _min_val, _rng);
    m->header->eb *= _rng;
    m->header->logging_max = _max_val;
    m->header->logging_min = _min_val;
  }

  auto c = (psz::CompressorF8*)m->compressor;
  c->compress(m, IN_d_data, OUT_dptr_compressed, OUT_compressed_bytes, m->stream);
  c->export_header(*OUT_compressed_metadata);

  return 0;
}

int psz_decompress_float(
    psz_resource* m, uint8_t* IN_d_compressed, size_t const IN_compressed_len,
    float* OUT_d_decompressed)
{
  auto c = (psz::CompressorF4*)m->compressor;
  c->decompress(m->header, IN_d_compressed, OUT_d_decompressed, m->stream);
  return 0;
}

int psz_decompress_double(
    psz_resource* m, uint8_t* IN_d_compressed, size_t const IN_compressed_len,
    double* OUT_d_decompressed)
{
  auto c = (psz::CompressorF8*)m->compressor;
  c->decompress(m->header, IN_d_compressed, OUT_d_decompressed, m->stream);
  return 0;
}