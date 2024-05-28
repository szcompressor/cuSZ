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
#include "mem.hh"
#include "typing.hh"

namespace psz {

using namespace portable;

// extra helper
struct CompressorHelper {
  static int autotune_phf_coarse(psz_context* ctx);
};

template <class C>
class Compressor {
 private:
  struct impl;
  std::unique_ptr<impl> pimpl;

 public:
  using BYTE = uint8_t;
  using T = typename C::T;
  using E = typename C::E;

  using TimeRecord = std::vector<std::tuple<const char*, double>>;
  using timerecord_t = TimeRecord*;

 public:
  Compressor();
  ~Compressor();

  template <class CONFIG>
  Compressor* init(CONFIG* config, bool iscomp = true, bool dbg = false);
  Compressor* compress(pszctx*, T*, BYTE**, size_t*, psz_stream_t);
  Compressor* decompress(psz_header*, BYTE*, T*, psz_stream_t);
  Compressor* clear_buffer();

  // getter
  Compressor* export_header(psz_header&);
  Compressor* export_header(psz_header*);
  Compressor* export_timerecord(TimeRecord*);
};

}  // namespace psz

#endif /* C3E05282_5791_4E76_9D49_EC31A316EC29 */
