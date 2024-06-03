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
#include "port.hh"
#include "tehm.hh"

psz_compressor* capi_psz_create(
    psz_dtype const dtype, psz_len3 const uncomp_len3,
    psz_predtype const predictor, int const quantizer_radius,
    psz_codectype const codec)
{
  auto comp = new psz_compressor{
      .compressor = nullptr,
      .ctx = pszctx_minimal_workset(dtype, predictor, quantizer_radius, codec),
      .header = new psz_header,  // TODO link to comp->header
      .type = dtype,
      .last_error = CUSZ_SUCCESS};

  pszctx_set_len(comp->ctx, uncomp_len3);
  capi_phf_coarse_tune(
      comp->ctx->data_len, &comp->ctx->vle_sublen, &comp->ctx->vle_pardeg);
  if (comp->type == F4 or comp->type == F8)
    comp->compressor = comp->type == F4
                           ? (void*)(new psz::CompressorF4(comp->ctx))
                           : (void*)(new psz::CompressorF8(comp->ctx));
  else
    comp->last_error = PSZ_TYPE_UNSUPPORTED;

  return comp;
}

psz_compressor* capi_psz_create_default(
    psz_dtype const dtype, psz_len3 const uncomp_len3)
{
  auto comp = new psz_compressor{
      .compressor = nullptr,
      .ctx = pszctx_default_values(),
      .header = new psz_header,  // TODO link to ->header
      .type = dtype,
      .last_error = CUSZ_SUCCESS};

  pszctx_set_len(comp->ctx, uncomp_len3);
  capi_phf_coarse_tune(
      comp->ctx->data_len, &comp->ctx->vle_sublen, &comp->ctx->vle_pardeg);
  if (comp->type == F4 or comp->type == F8)
    comp->compressor = comp->type == F4
                           ? (void*)(new psz::CompressorF4(comp->ctx))
                           : (void*)(new psz::CompressorF8(comp->ctx));
  else
    comp->last_error = PSZ_TYPE_UNSUPPORTED;

  return comp;
}

psz_compressor* capi_psz_create_from_context(
    pszctx* const ctx, psz_len3 uncomp_len3)
{
  auto comp = new psz_compressor{
      .compressor = nullptr,
      .ctx = ctx,
      .header = nullptr,
      .type = ctx->dtype,
      .last_error = CUSZ_SUCCESS};

  pszctx_set_len(comp->ctx, uncomp_len3);
  capi_phf_coarse_tune(
      comp->ctx->data_len, &comp->ctx->vle_sublen, &comp->ctx->vle_pardeg);
  if (comp->type == F4 or comp->type == F8)
    comp->compressor = comp->type == F4
                           ? (void*)(new psz::CompressorF4(comp->ctx))
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
  ctx->radius = h->radius;
  ctx->vle_pardeg = h->vle_pardeg;
  ctx->radius = h->radius;
  ctx->dict_size = h->radius * 2;
  ctx->x = h->x, ctx->y = h->y, ctx->z = h->z;

  comp->header = h;

  if (comp->type == F4 or comp->type == F8)
    comp->compressor = comp->type == F4
                           ? (void*)(new psz::CompressorF4(comp->ctx))
                           : (void*)(new psz::CompressorF8(comp->ctx));
  else
    comp->last_error = PSZ_TYPE_UNSUPPORTED;

  return comp;
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

pszerror capi_psz_compress(
    psz_compressor* comp, void* in, psz_len3 const uncomp_len3,
    double const eb, psz_mode const mode, uint8_t** comped, size_t* comp_bytes,
    psz_header* header, void* record, void* stream)
{
  comp->ctx->eb = eb;
  comp->ctx->mode = mode;

  if (comp->type == F4) {
    auto cor = (psz::CompressorF4*)(comp->compressor);

    cor->compress(comp->ctx, (f4*)(in), comped, comp_bytes, stream);
    cor->export_header(*header);
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

pszerror capi_psz_decompress(
    psz_compressor* comp, uint8_t* comped, size_t const comp_len,
    void* decomped, psz_len3 const decomp_len, void* record, void* stream)
{
  if (comp->type == F4) {
    auto cor = (psz::CompressorF4*)(comp->compressor);

    cor->decompress(comp->header, comped, (f4*)(decomped), (GpuStreamT)stream);
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



