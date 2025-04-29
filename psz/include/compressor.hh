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

#ifndef PSZ_COMPRESSOR_HH
#define PSZ_COMPRESSOR_HH

#include <memory>

#include "busyheader.hh"
#include "compbuf.hh"
#include "context.h"
#include "cusz/type.h"
#include "header.h"
#include "hf_hl.hh"
#include "typing.hh"

namespace psz {

template <typename DType>
class
    /*[[deprecated("use non-OOD compression pieline instead")]]*/ Compressor {
 private:
  // struct impl;
  // std::unique_ptr<impl> pimpl;

 public:
  using T = DType;
  using E = u2;
  using FP = T;
  using M = u4;
  using BYTE = u1;
  using B = u1;
  using H = u4;
  using H4 = u4;
  using H8 = u8;

  using Buf = CompressorBuffer<DType>;

  // encapsulations
  int hist_generic_grid_dim;
  int hist_generic_block_dim;
  int hist_generic_shmem_use;
  int hist_generic_repeat;
  size_t len;
  BYTE* comp_codec_out{nullptr};
  size_t comp_codec_outlen{0};
  uint32_t nbyte[END];
  float time_sp;
  double eb, eb_r, ebx2, ebx2_r;
  psz_header* const header_ref;

  Buf* mem;
  phf::Buf<E>* buf_hf;

 private:
  void compress_data_processing(pszctx* ctx, T* in, void* stream);
  void compress_merge_update_header(pszctx* ctx, BYTE** out, size_t* outlen, void* stream);

 public:
  // comp, ctor(...) + no further init
  Compressor(psz_context*);
  Compressor(psz_header*);
  // dtor, releasing
  ~Compressor();

  void compress(pszctx*, T*, BYTE**, size_t*, psz_stream_t);
  void decompress(psz_header*, BYTE*, T*, psz_stream_t);
  void dump_compress_intermediate(pszctx*, psz_stream_t);
  void clear_buffer();

  // getter
  void export_header(psz_header&);
  // void export_timerecord(TimeRecord*);
  // void export_timerecord(float*);
};

using TimeRecordTuple = std::tuple<const char*, double>;
using TimeRecord = std::vector<TimeRecordTuple>;
using timerecord_t = TimeRecord*;

using CompressorF4 = Compressor<f4>;
using CompressorF8 = Compressor<f8>;

}  // namespace psz

template <typename T, typename E>
using psz_buf = psz::CompressorBuffer<T, E>;

#define PSZ_BUF psz_buf<T, E>

namespace psz {

template <typename T, typename E>
struct compression_pipeline {
  static void* compress_init(psz_context* ctx);
  static void* decompress_init(psz_header* header);
  static void compress(pszctx*, PSZ_BUF* mem, T*, u1**, size_t*, psz_stream_t);
  static void decompress(psz_header* header, PSZ_BUF* mem, u1* in, T* out, psz_stream_t stream);
  static void release(PSZ_BUF* mem);
  static void compress_dump_internal_buf(pszctx* ctx, PSZ_BUF* mem, psz_stream_t stream);
};

}  // namespace psz

#endif /* PSZ_COMPRESSOR_HH */
