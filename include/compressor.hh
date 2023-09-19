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
#include "hf/hf.hh"
#include "mem.hh"
#include "typing.hh"

namespace cusz {

// extra helper
struct CompressorHelper {
  static int autotune_coarse_parhf(cusz_context* ctx);
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
  using Header = cusz_header;

  using H = u4;
  using H4 = u4;
  using H8 = u8;

  using TimeRecord = std::vector<std::tuple<const char*, double>>;
  using timerecord_t = TimeRecord*;

 private:
  // profiling
  TimeRecord timerecord;

  // header
  Header header;

  // external codec that has standalone internals
  Codec* codec;

  float time_pred, time_hist, time_sp;

  // sizes
  dim3 len3;
  size_t len;
  int splen;

  // configs
  float outlier_density{0.2};

  // buffers

  pszmempool_cxx<T, E, H>* mem;

 public:
  Compressor() = default;
  ~Compressor();

  // public methods
  template <class CONFIG>
  Compressor* init(CONFIG* config, bool dbg_print = false);
  Compressor* compress(
      cusz_context*, T*, BYTE*&, size_t&, void* = nullptr, bool = false);
  Compressor* decompress(
      cusz_header*, BYTE*, T*, void* = nullptr, bool = true);
  Compressor* clear_buffer();
  Compressor* dump(std::vector<pszmem_dump>, char const*);
  Compressor* destroy();

  // getter
  Compressor* export_header(cusz_header&);
  Compressor* export_header(cusz_header*);
  Compressor* export_timerecord(TimeRecord*);

 private:
  // helper
  Compressor* collect_comp_time();
  Compressor* collect_decomp_time();
  Compressor* merge_subfiles(
      pszpredictor_type, T*, szt, BYTE*, szt, T*, M*, szt, void*);
};

}  // namespace cusz

#endif /* C3E05282_5791_4E76_9D49_EC31A316EC29 */
