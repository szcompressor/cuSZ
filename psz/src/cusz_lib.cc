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

#include <cstdint>

#include "busyheader.hh"
#include "compressor.hh"
#include "context.h"
#include "cusz.h"
#include "cusz/type.h"
#include "hfclass.hh"
#include "port.hh"
#include "tehm.hh"

pszcompressor* psz_create(
    psz_dtype const dtype, psz_predtype const predictor,
    int const quantizer_radius, psz_codectype const codec, double const eb,
    psz_mode const mode)
{
  return new pszcompressor{
      .compressor = dtype == F4 ? (void*)(new psz::CompressorF4())
                                : (void*)(new psz::CompressorF8()),
      .ctx =
          new pszctx{
              .dtype = dtype,
              .pred_type = predictor,
              .codec_type = codec,
              .mode = mode,
              .eb = eb,
              .dict_size = quantizer_radius * 2,
              .radius = quantizer_radius,
          },
      .header = new pszheader,  // TODO link to Compressor::header
      .type = dtype,
  };
}

pszcompressor* psz_create_default(
    psz_dtype const dtype, double const eb, psz_mode const mode)
{
  return psz_create(dtype, Lorenzo, 512, Huffman, eb, mode);
}

pszcompressor* psz_create_from_context(pszctx* const ctx)
{
  return new pszcompressor{
      .compressor = ctx->dtype == F4 ? (void*)(new psz::CompressorF4())
                                     : (void*)(new psz::CompressorF8()),
      .ctx = ctx,
      .header = new pszheader,
      .type = ctx->dtype,
  };
}

pszerror psz_release(pszcompressor* comp)
{
  if (comp->type == F4)
    delete (psz::CompressorF4*)comp->compressor;
  else
    throw std::runtime_error("Type is not supported.");

  delete comp;
  return CUSZ_SUCCESS;
}

pszerror psz_compress_init(pszcompressor* comp, psz_len3 const uncomp_len)
{
  pszctx_set_len(comp->ctx, uncomp_len);

  // Be cautious of autotuning! The default value of pardeg is not robust.
  psz::CompressorHelper::autotune_phf_coarse(comp->ctx);

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

pszerror psz_compress(
    pszcompressor* comp, void* in, psz_len3 const uncomp_len,
    uint8_t** compressed, size_t* comp_bytes, pszheader* header, void* record,
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

pszerror psz_decompress_init(pszcompressor* comp, pszheader* header)
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

pszerror psz_decompress(
    pszcompressor* comp, uint8_t* compressed, size_t const comp_len,
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

pszerror psz_decompress_init_v2(pszcompressor* comp, pszheader* header)
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
