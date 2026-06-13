#ifndef PSZ_COMPRESSOR_HH
#define PSZ_COMPRESSOR_HH

#include "cusz/context.h"
#include "cusz/header.h"
#include "cusz/type.h"
#include "mem/buf_comp.hh"

template <typename T, typename E>
using psz_buf = psz::Buf_Comp<T, E>;

#define PSZ_BUF psz_buf<T, E>

namespace psz {

template <typename T, typename E>
struct compression_pipeline {
  static void* compress_init(psz_ctx* ctx);
  static void* decompress_init(psz_header* header);
  static int compress(psz_ctx*, PSZ_BUF* mem, T*, u1**, size_t*, psz_stream_t);
  static int compress_analysis(psz_ctx*, PSZ_BUF* mem, T*, u4*, psz_stream_t);
  static int decompress(psz_header* header, PSZ_BUF* mem, u1* in, T* out, psz_stream_t stream);
  static void release(PSZ_BUF* mem);
  static void compress_dump_internal_buf(psz_ctx* ctx, PSZ_BUF* mem, psz_stream_t stream);
};

}  // namespace psz

#endif /* PSZ_COMPRESSOR_HH */
