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
  // skip_hf: drop the Huffman and packed-bytes buffers, for a caller that only
  // runs comp_predict / decomp_predict and never reaches an encoded bitstream
  static void* compress_init(psz_ctx* ctx, bool skip_hf = false);
  static void* decompress_init(psz_header* header);
  static int compress(psz_ctx*, PSZ_BUF* mem, T*, u1**, size_t*, psz_stream_t);
  static int compress_analysis(psz_ctx*, PSZ_BUF* mem, T*, u4*, psz_stream_t);
  static int decompress(
      psz_header* header, PSZ_BUF* mem, u1* in, T* out, psz_stream_t stream,
      bool use_hfd_coarse = false);
  static void release(PSZ_BUF* mem);
  static void compress_dump_internal_buf(psz_ctx* ctx, PSZ_BUF* mem, psz_stream_t stream);

  static int comp_predict(
      psz_ctx* ctx, PSZ_BUF* mem, T* in, psz_stream_t stream, bool force_global = false);
  static void decomp_scatter(
      psz_header* header, _ptb::compact_cell<T, M>* d_spval_idx, T* d_space, psz_stream_t stream);
  static void decomp_predict(
      psz_header* header, PSZ_BUF* mem, T* d_anchor, T* d_xdata, psz_stream_t stream);
};

}  // namespace psz

#endif /* PSZ_COMPRESSOR_HH */
