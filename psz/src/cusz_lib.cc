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
#include "hfclass.hh"
#include "port.hh"
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

pszcompressor* psz_create(pszframe* _framework, psz_dtype _type)
{
  auto comp = new pszcompressor{.framework = _framework, .type = _type};

  if (comp->type == F4)
    comp->compressor = new psz::CompressorF4();
  else
    throw std::runtime_error("Type is not supported.");

  return comp;
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

// TODO config is redundant when it comes to CLI
pszerror psz_compress_init(
    pszcompressor* comp, pszlen const uncomp_len, pszctx* ctx)
{
  comp->ctx = ctx;
  pszctx_set_len(comp->ctx, uncomp_len);

  // Be cautious of autotuning! The default value of pardeg is not robust.
  psz::CompressorHelper::autotune_phf_coarse(comp->ctx);

  if (comp->type == F4) {
    auto cor = (psz::CompressorF4*)(comp->compressor);
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
  if (comp->type == F4) {
    auto cor = (psz::CompressorF4*)(comp->compressor);
    cor->init(header, false);
  }
  else {
    throw std::runtime_error(
        std::string(__FUNCTION__) + ": Type is not supported.");
  }

  return CUSZ_SUCCESS;
}

pszerror psz_decompress(
    pszcompressor* comp, pszout compressed, size_t const comp_len,
    void* decompressed, pszlen const decomp_len, void* record, void* stream)
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
