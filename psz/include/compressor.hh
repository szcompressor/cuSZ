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

#include <memory>

#include "busyheader.hh"
#include "context.h"
#include "cusz/type.h"
#include "fzg_class.hh"
#include "header.h"
#include "hf_ood.hh"
#include "typing.hh"

namespace psz {

template <typename DType>
class Compressor {
 private:
  struct impl;
  std::unique_ptr<impl> pimpl;

 public:
  using T = DType;
  using E = uint16_t;
  using BYTE = uint8_t;

  psz_header* const header_ref;

  using TimeRecord = std::vector<std::tuple<const char*, double>>;
  using timerecord_t = TimeRecord*;

 public:
  // comp, ctor(...) + no further init
  Compressor(psz_context*, bool debug = false);
  Compressor(psz_header*, bool debug = false);
  // dtor, releasing
  ~Compressor();

  void compress(pszctx*, T*, BYTE**, size_t*, psz_stream_t);
  void decompress(psz_header*, BYTE*, T*, psz_stream_t);
  void dump_compress_intermediate(pszctx*, psz_stream_t);
  void clear_buffer();

  // getter
  void export_header(psz_header&);
  void export_timerecord(TimeRecord*);
  void export_timerecord(float*);
};

using TimeRecordTuple = std::tuple<const char*, double>;
using TimeRecord = std::vector<TimeRecordTuple>;
using timerecord_t = TimeRecord*;

template <typename Input, bool Fast = true>
struct CompressorInternalTypes {
 public:
  using T = Input;
  using E2 = uint16_t;
  using E1 = uint8_t;
  using FP = T;
  using M = uint32_t;

  /* lossless codec */

  // TODO: runtime switch
  using CodecHF_U2 = phf::HuffmanCodec<E2>;
  using CodecHF_U1 = phf::HuffmanCodec<E1>;

  // The input is mandetory to be u2.
  using CodecFZG = psz::FzgCodec;
};

using CompressorF4 = Compressor<f4>;
using CompressorF8 = Compressor<f8>;

}  // namespace psz

#endif /* C3E05282_5791_4E76_9D49_EC31A316EC29 */
