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
#include "context.h"
#include "cusz.h"
#include "cusz/type.h"
#include "mem/cxx_memobj.h"
#include "tehm.hh"

using _portable::memobj;
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
      .header = new psz_header,  // TODO link to comp->header
      .type = dtype,
      .last_error = CUSZ_SUCCESS};

  pszctx_set_len(comp->ctx, uncomp_len3);
  capi_phf_coarse_tune(comp->ctx->data_len, &comp->ctx->vle_sublen, &comp->ctx->vle_pardeg);
  if (comp->type == F4 or comp->type == F8)
    comp->compressor = comp->type == F4 ? (void*)(new psz::CompressorF4(comp->ctx))
                                        : (void*)(new psz::CompressorF8(comp->ctx));
  else
    comp->last_error = PSZ_TYPE_UNSUPPORTED;

  return comp;
}

void* capi_psz_create__experimental(
    psz_dtype const dtype, psz_len3 const uncomp_len3, psz_predtype const predictor,
    int const quantizer_radius, psz_codectype const codec)
{
  return (void*)capi_psz_create(dtype, uncomp_len3, predictor, quantizer_radius, codec);
}

psz_compressor* capi_psz_create_default(psz_dtype const dtype, psz_len3 const uncomp_len3)
{
  auto comp = new psz_compressor{
      .compressor = nullptr,
      .ctx = pszctx_default_values(),
      .header = new psz_header,  // TODO link to ->header
      .type = dtype,
      .last_error = CUSZ_SUCCESS};

  pszctx_set_len(comp->ctx, uncomp_len3);
  capi_phf_coarse_tune(comp->ctx->data_len, &comp->ctx->vle_sublen, &comp->ctx->vle_pardeg);
  if (comp->type == F4 or comp->type == F8)
    comp->compressor = comp->type == F4 ? (void*)(new psz::CompressorF4(comp->ctx))
                                        : (void*)(new psz::CompressorF8(comp->ctx));
  else
    comp->last_error = PSZ_TYPE_UNSUPPORTED;

  return comp;
}

psz_compressor* capi_psz_create_from_context(pszctx* const ctx, psz_len3 uncomp_len3)
{
  auto comp = new psz_compressor{
      .compressor = nullptr,
      .ctx = ctx,
      .header = nullptr,
      .type = ctx->dtype,
      .last_error = CUSZ_SUCCESS};

  pszctx_set_len(comp->ctx, uncomp_len3);
  capi_phf_coarse_tune(comp->ctx->data_len, &comp->ctx->vle_sublen, &comp->ctx->vle_pardeg);
  if (comp->type == F4 or comp->type == F8)
    comp->compressor = comp->type == F4 ? (void*)(new psz::CompressorF4(comp->ctx))
                                        : (void*)(new psz::CompressorF8(comp->ctx));
  else
    comp->last_error = PSZ_TYPE_UNSUPPORTED;

  return comp;
}

psz_compressor* capi_psz_create_from_header(psz_header* const h)
{
  auto comp = new psz_compressor{
      .compressor = nullptr,
      .ctx = pszctx_default_values(),
      .header = nullptr,
      .type = h->dtype,
      .last_error = CUSZ_SUCCESS};

  auto ctx = comp->ctx;
  ctx->vle_pardeg = h->vle_pardeg;
  ctx->radius = h->radius;
  ctx->dict_size = h->radius * 2;
  ctx->x = h->x, ctx->y = h->y, ctx->z = h->z;

  comp->header = h;

  // TODO remove extra header-ctx conversion and states
  comp->ctx->codec1_type = h->codec1_type;

  if (comp->type == F4 or comp->type == F8)
    comp->compressor = comp->type == F4 ? (void*)(new psz::CompressorF4(comp->ctx))
                                        : (void*)(new psz::CompressorF8(comp->ctx));
  else
    comp->last_error = PSZ_TYPE_UNSUPPORTED;

  return comp;
}

void* capi_psz_create_from_header__experimental(psz_header* const h)
{
  return (void*)capi_psz_create_from_header(h);
}

pszerror capi_psz_release(psz_compressor* comp)
{
  if (comp->type == F4)
    delete (psz::CompressorF4*)comp->compressor;
  else if (comp->type == F8)
    delete (psz::CompressorF8*)comp->compressor;
  else
    return PSZ_TYPE_UNSUPPORTED;

  delete comp;
  return CUSZ_SUCCESS;
}

pszerror capi_psz_release__experimental(void* comp)
{
  return capi_psz_release((psz_compressor*)comp);
}

pszerror capi_psz_compress(
    psz_compressor* comp, void* d_in, psz_len3 const in_len3, double const eb, psz_mode const mode,
    uint8_t** comped, size_t* comp_bytes, psz_header* header, void* record, void* stream)
{
  comp->ctx->eb = eb;
  comp->ctx->mode = mode;
  comp->ctx->user_input_eb = eb;

  if (mode == Rel) {
    double _max, _min, _rng;
    if (comp->type == F4) {
      auto _input = memobj<f4>(comp->ctx->x, comp->ctx->y, comp->ctx->z, "capi_data_input");
      _input.dptr((f4*)d_in);
      _input.extrema_scan(_max, _min, _rng);
    }
    else {
      printf("[psz::error] non-F4 is not currently supported.");
    }
    comp->ctx->eb *= _rng;
    comp->ctx->logging_max = _max;
    comp->ctx->logging_min = _min;
  }

  if (comp->type == F4) {
    auto cor = (psz::CompressorF4*)(comp->compressor);

    cor->compress(comp->ctx, (f4*)(d_in), comped, comp_bytes, stream);
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

  header->user_input_eb = comp->ctx->user_input_eb;
  header->logging_max = comp->ctx->logging_max;
  header->logging_min = comp->ctx->logging_min;
  header->logging_mode = mode;
  header->pred_type = comp->ctx->pred_type;

  return CUSZ_SUCCESS;
}

pszerror capi_psz_compress__experimental(
    void* comp, void* d_in, psz_len3 const in_len3, double const eb, psz_mode const mode,
    uint8_t** d_compressed, size_t* comp_bytes, psz_header* header, void* record, void* stream)
{
  return capi_psz_compress(
      (psz_compressor*)comp, d_in, in_len3, eb, mode, d_compressed, comp_bytes, header, record,
      stream);
}

pszerror capi_psz_decompress(
    psz_compressor* comp, uint8_t* comped, size_t const comp_len, void* decomped,
    psz_len3 const decomp_len, void* record, void* stream)
{
  if (comp->type == F4) {
    auto cor = (psz::CompressorF4*)(comp->compressor);

    cor->decompress(comp->header, comped, (f4*)(decomped), (cudaStream_t)stream);
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
  if (comp->type == F4) {
    auto cor = (psz::CompressorF4*)(comp->compressor);
    cor->clear_buffer();
  }
  else {
    // TODO put to log-queue
    cerr << std::string(__FUNCTION__) + ": Type is not supported." << endl;
    return PSZ_TYPE_UNSUPPORTED;
  }

  return CUSZ_SUCCESS;
}
