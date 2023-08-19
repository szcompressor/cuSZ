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

#include "busyheader.hh"
#include "compressor.hh"
#include "context.h"
#include "cusz.h"
#include "cusz/type.h"
#include "hf/hf.hh"
#include "tehm.hh"

pszpredictor pszdefault_predictor() { return {Lorenzo}; }
pszquantizer pszdefault_quantizer() { return {512}; }
pszhfrc pszdefault_hfcoder() { return {Sword, Coarse, 1024, 768}; }
pszframe* pszdefault_framework()
{
  return new pszframe{
      pszdefault_predictor(), pszdefault_quantizer(), pszdefault_hfcoder(),
      20};
}

pszcompressor* psz_create(pszframe* _framework, pszdtype _type)
{
  auto comp = new pszcompressor{.framework = _framework, .type = _type};

  if (comp->type == F4) {
    using Compressor = cusz::CompressorF4;
    comp->compressor = new Compressor();
  }
  else {
    throw std::runtime_error("Type is not supported.");
  }

  return comp;
}

pszerror psz_release(pszcompressor* comp)
{
  delete comp;
  return CUSZ_SUCCESS;
}

pszerror psz_compress_init(
    pszcompressor* comp, pszlen const uncomp_len, pszrc* config)
{
  comp->ctx = new pszctx;
  pszctx_set_len(comp->ctx, uncomp_len);
  comp->ctx->eb = config->eb;
  comp->ctx->mode = config->mode;
  // Be cautious of autotuning! The default value of pardeg is not robust.
  cusz::CompressorHelper::autotune_coarse_parhf(comp->ctx);

  if (comp->type == F4) {
    auto cor = (cusz::CompressorF4*)(comp->compressor);
    cor->init(comp->ctx);
  }
  else {
    throw std::runtime_error(
        std::string(__FUNCTION__) + ": Type is not supported.");
  }

  return CUSZ_SUCCESS;
}

pszerror psz_compress(
    pszcompressor* comp, void* in, pszlen const uncomp_len,
    ptr_pszout compressed, size_t* comp_bytes, pszheader* header, void* record,
    cudaStream_t stream)
{
  if (comp->type == F4) {
    auto cor = (cusz::CompressorF4*)(comp->compressor);

    cor->compress(comp->ctx, (f4*)(in), *compressed, *comp_bytes, stream);
    cor->export_header(*header);
    cor->export_timerecord((cusz::TimeRecord*)record);
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
  if (comp->type == F4) {
    auto cor = (cusz::CompressorF4*)(comp->compressor);
    cor->init(header);
  }
  else {
    throw std::runtime_error(
        std::string(__FUNCTION__) + ": Type is not supported.");
  }

  return CUSZ_SUCCESS;
}

pszerror psz_decompress(
    pszcompressor* comp, pszout compressed, size_t const comp_len,
    void* decompressed, pszlen const decomp_len, void* record,
    cudaStream_t stream)
{
  if (comp->type == F4) {
    auto cor = (cusz::CompressorF4*)(comp->compressor);

    cor->decompress(comp->header, compressed, (f4*)(decompressed), stream);
    cor->export_timerecord((cusz::TimeRecord*)record);
  }
  else {
    throw std::runtime_error(
        std::string(__FUNCTION__) + ": Type is not supported.");
  }

  return CUSZ_SUCCESS;
}
