/**
 * @file compressor.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-23
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef C3E05282_5791_4E76_9D49_EC31A316EC29
#define C3E05282_5791_4E76_9D49_EC31A316EC29

#include "busyheader.hh"
#include "context.h"
#include "cusz/type.h"
#include "header.h"
#include "hfclass.hh"
#include "mem.hh"
#include "module/cxx_module.hh"
#include "typing.hh"

namespace cusz {

// extra helper
struct CompressorHelper {
  static int autotune_phf_coarse(psz_context* ctx);
};

template <class TEHM>
class Compressor {
 public:
  using Codec = typename TEHM::Codec;
  using BYTE = uint8_t;
  using B = u1;

  using T = typename TEHM::T;
  using E = typename TEHM::E;
  using FP = typename TEHM::FP;
  using M = typename TEHM::M;
  using Header = pszheader;

  using H = u4;
  using H4 = u4;
  using H8 = u8;

  using TimeRecord = std::vector<std::tuple<const char*, double>>;
  using timerecord_t = TimeRecord*;

 private:
  // profiling
  TimeRecord timerecord;

  std::vector<pszerror> error_list;

  // header
  Header header;

  // external codec that has standalone internals
  Codec* codec;

  float time_pred, time_hist, time_sp;

  size_t len;
  int splen;

  BYTE* comp_hf_out{nullptr};
  size_t comp_hf_outlen{0};

  // configs
  float outlier_density{0.2};

  // buffers
  uint32_t nbyte[Header::END];

 public:
  pszmempool_cxx<T, E, H>* mem;
  pszcompact_cxx<T>* _2403_compact;

 public:
  Compressor(){};
  ~Compressor();

  // public methods
  template <class CONFIG>
  Compressor* init(CONFIG* config, bool iscompression=true, bool dbg_print = false);
  Compressor* compress(pszctx*, T*, BYTE**, size_t*, uninit_stream_t);
  Compressor* compress_predict(pszctx*, T*, uninit_stream_t);
  Compressor* compress_encode(pszctx*, uninit_stream_t);
  Compressor* compress_encode_use_prebuilt(pszctx*, uninit_stream_t);
  Compressor* compress_merge_update_header(pszctx*, BYTE**, szt*, void*);
  Compressor* compress_collect_kerneltime();

  Compressor* decompress(pszheader*, BYTE*, T*, uninit_stream_t);
  Compressor* decompress_scatter(pszheader*, BYTE*, T*, uninit_stream_t);
  Compressor* decompress_decode(pszheader*, BYTE*, uninit_stream_t);
  Compressor* decompress_predict(pszheader*, BYTE*, T*, T*, uninit_stream_t);
  Compressor* decompress_collect_kerneltime();

  Compressor* clear_buffer();
  Compressor* optional_dump(pszctx*, pszmem_dump const);

  // getter
  Compressor* export_header(pszheader&);
  Compressor* export_header(pszheader*);
  Compressor* export_timerecord(TimeRecord*);
};

}  // namespace cusz

#endif /* C3E05282_5791_4E76_9D49_EC31A316EC29 */
