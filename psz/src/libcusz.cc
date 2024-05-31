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
#include "port.hh"
#include "tehm.hh"

psz_compressor* capi_psz_create(
    psz_dtype const dtype, psz_predtype const predictor,
    int const quantizer_radius, psz_codectype const codec, double const eb,
    psz_mode const mode)
{
  return new psz_compressor{
      .compressor = dtype == F4 ? (void*)(new psz::CompressorF4())
                                : (void*)(new psz::CompressorF8()),
      .ctx = pszctx_minimal_working_set(
          dtype, predictor, quantizer_radius, codec, eb, mode),
      .header = new psz_header,  // TODO link to compressor->header
      .type = dtype,
  };
}

psz_compressor* capi_psz_create_default(
    psz_dtype const dtype, double const eb, psz_mode const mode)
{
  return new psz_compressor{
      .compressor = dtype == F4 ? (void*)(new psz::CompressorF4())
                                : (void*)(new psz::CompressorF8()),
      .ctx = pszctx_default_values(),
      .header = new psz_header,  // TODO link to compressor->header
      .type = dtype,
  };
}

psz_compressor* capi_psz_create_from_context(pszctx* const ctx)
{
  return new psz_compressor{
      .compressor = ctx->dtype == F4 ? (void*)(new psz::CompressorF4())
                                     : (void*)(new psz::CompressorF8()),
      .ctx = ctx,
      .header = new psz_header,
      .type = ctx->dtype,
  };
}

pszerror capi_psz_release(psz_compressor* comp)
{
  if (comp->type == F4)
    delete (psz::CompressorF4*)comp->compressor;
  else
    throw std::runtime_error("Type is not supported.");

  delete comp;
  return CUSZ_SUCCESS;
}

pszerror capi_psz_compress_init(
    psz_compressor* comp, psz_len3 const uncomp_len)
{
  pszctx_set_len(comp->ctx, uncomp_len);

  // Be cautious of autotuning! The default value of pardeg is not robust.
  capi_phf_coarse_tune(
      comp->ctx->data_len, &comp->ctx->vle_sublen, &comp->ctx->vle_pardeg);

  if (comp->type == F4)
    static_cast<psz::CompressorF4*>(comp->compressor)->init(comp->ctx);
  else if (comp->type == F8)
    static_cast<psz::CompressorF8*>(comp->compressor)->init(comp->ctx);
  else {
    throw std::runtime_error(
        std::string(__FUNCTION__) + ": Type is not supported.");
  }

  return CUSZ_SUCCESS;
}

pszerror capi_psz_compress(
    psz_compressor* comp, void* in, psz_len3 const uncomp_len,
    uint8_t** compressed, size_t* comp_bytes, psz_header* header, void* record,
    void* stream)
{
  if (comp->type == F4) {
    auto cor = (psz::CompressorF4*)(comp->compressor);

    cor->compress(comp->ctx, (f4*)(in), compressed, comp_bytes, stream);
    cor->export_header(*header);
    cor->export_timerecord((psz::TimeRecord*)record);
  }
  else {
    throw std::runtime_error(
        std::string(__FUNCTION__) + ": Type is not supported.");
  }

  return CUSZ_SUCCESS;
}

pszerror capi_psz_decompress_init(psz_compressor* comp, psz_header* header)
{
  comp->header = header;
  if (comp->type == F4)
    static_cast<psz::CompressorF4*>(comp->compressor)->init(header, false);
  else if (comp->type == F8)
    static_cast<psz::CompressorF8*>(comp->compressor)->init(header, false);
  else {
    throw std::runtime_error(
        std::string(__FUNCTION__) + ": Type is not supported.");
  }

  return CUSZ_SUCCESS;
}

pszerror capi_psz_decompress(
    psz_compressor* comp, uint8_t* compressed, size_t const comp_len,
    void* decompressed, psz_len3 const decomp_len, void* record, void* stream)
{
  if (comp->type == F4) {
    auto cor = (psz::CompressorF4*)(comp->compressor);

    cor->decompress(
        comp->header, compressed, (f4*)(decompressed), (GpuStreamT)stream);
    cor->export_timerecord((psz::TimeRecord*)record);
  }
  else {
    throw std::runtime_error(
        std::string(__FUNCTION__) + ": Type is not supported.");
  }

  return CUSZ_SUCCESS;
}
