#ifndef PSZ_COMPRESSOR_HH
#define PSZ_COMPRESSOR_HH

#include "cusz/header.h"
#include "cusz/type.h"
#include "mem/buf_comp.hh"

template <typename T>
using psz_buf = psz::Buf_Comp<T>;

#define PSZ_BUF psz_buf<T>

namespace psz {

template <typename T, typename E, class PPL = void>
struct compression_pipeline {
  static void* compress_init(
      psz_ctx* ctx, u1* archive = nullptr, int nstage = 2, bool eq4 = true);
  static void* decompress_init(psz_header* header, int nstage = 2, bool eq4 = true);
  static int compress(psz_ctx*, PSZ_BUF* mem, T*, u1**, size_t*, psz_stream_t);
  static int compress_analysis(psz_ctx*, PSZ_BUF* mem, T*, u4*, psz_stream_t);
  static int decompress(
      psz_header* header, PSZ_BUF* mem, u1* in, T* out, psz_stream_t stream,
      bool use_hfd_coarse = false);
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
