#ifndef D776A673_682E_40DB_BA19_2E9A516D44CC
#define D776A673_682E_40DB_BA19_2E9A516D44CC

#include "busyheader.hh"
#include "compressor.hh"
#include "context.h"
#include "cusz/type.h"
#include "header.h"
#include "tehm.hh"

template <typename T = f4>
class psz_testframe {
  using Compressor = cusz::Compressor<cusz::TEHM<f4>>;
  using BYTE = u1;

 public:
  static void full_compress(
      pszctx* ctx, Compressor* cor, T* in, BYTE** out, szt* outlen,
      uninit_stream_t stream);

  static void full_decompress(
      pszheader* header, Compressor* cor, u1* d_compressed, T* out,
      uninit_stream_t stream);

  static void pred_comp_decomp(
      pszctx* ctx, Compressor* cor, T* in, T* out, uninit_stream_t stream);

  static void pred_hist_comp(
      pszctx* ctx, Compressor* cor, T* in, uninit_stream_t stream,
      bool skip_print = false);

  static void pred_hist_hf_comp(
      pszctx* ctx, Compressor* cor, T* in, uninit_stream_t stream);
};

#endif /* D776A673_682E_40DB_BA19_2E9A516D44CC */
