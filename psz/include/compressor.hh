#ifndef PSZ_COMPRESSOR_HH
#define PSZ_COMPRESSOR_HH

#include "cusz/header.h"
#include "cusz/type.h"
#include "mem/buf_comp.hh"

template <typename T>
using psz_buf = psz::Buf_Comp<T>;

namespace psz {

template <typename T, class PPL = void>
struct compressor_cpp {
  using buf_t = Buf_Comp<T>;
  using stream_t = psz_stream_t;
  using header_t = psz_header;
  using ppl_t = psz_ppl;
  using ctx_t = psz_ctx;

  static void* compress_init(ctx_t*);
  static void* compress_init_2stage(ctx_t*);
  static void* compress_init_3stage(ctx_t*);
  static psz_data_summary compress_extrema(ctx_t*, T* in, stream_t);
  static int compress_process(ctx_t*, ppl_t, buf_t*, T* in, stream_t);
  static int compress_archive(ctx_t*, buf_t*, header_t*, u1** out, size_t* out_bytes, stream_t);
  static int compress_reset(ctx_t*, buf_t*, stream_t);
  static int compress_analysis(ctx_t*, buf_t*, T*, u4*, stream_t);

  static void* decompress_init(header_t*);
  static int decompress_process(ctx_t*, buf_t*, u1* in, T* out, stream_t);
  static int decompress_reset(ctx_t*, buf_t*, stream_t);

 private:
  struct dispatch;
  static void compress_dump_frame(ctx_t*, buf_t*, stream_t);
};

}  // namespace psz

#endif /* PSZ_COMPRESSOR_HH */
