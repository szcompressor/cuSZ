#ifndef D776A673_682E_40DB_BA19_2E9A516D44CC
#define D776A673_682E_40DB_BA19_2E9A516D44CC

#include "compressor.hh"
#include "context.h"
#include "cusz/type.h"
#include "header.h"
#include "tehm.hh"

template <typename T = f4>
class psz_testframe {
  using Compressor = cusz::Compressor<cusz::CompoundType<f4>>;
  using BYTE = u1;

 public:
  static void full_compress(
      pszctx* ctx, Compressor* cor, T* in, BYTE** out, szt* outlen,
      psz_stream_t stream);

  static void full_decompress(
      psz_header* header, Compressor* cor, u1* d_compressed, T* out,
      psz_stream_t stream);

  static void pred_comp_decomp(
      pszctx* ctx, Compressor* cor, T* in, T* out, psz_stream_t stream);

  static void pred_hist_comp(
      pszctx* ctx, Compressor* cor, T* in, psz_stream_t stream,
      bool skip_print = false);

  static void pred_hist_hf_comp(
      pszctx* ctx, Compressor* cor, T* in, psz_stream_t stream);
};

#endif /* D776A673_682E_40DB_BA19_2E9A516D44CC */
