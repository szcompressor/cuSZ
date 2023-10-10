#include "pipeline/testframe.hh"

#include "busyheader.hh"
#include "compressor.hh"
#include "context.h"
#include "hf/hf.hh"
#include "kernel.hh"
#include "mem.hh"
#include "port.hh"
#include "stat.hh"
// #include "utils.hh"
#include "utils/viewer.hh"

#define TESTFRAME       \
  template <typename T> \
  void psz_testframe<T>

TESTFRAME::full_compress(
    pszctx* ctx, Compressor* cor, T* in, BYTE** out, szt* outlen,
    uninit_stream_t stream)
{
  cor->compress_predict(ctx, in, stream);
  cor->compress_histogram(ctx, stream);
  cor->compress_encode(ctx, stream);
  cor->compress_merge(ctx, stream);
  cor->compress_update_header(ctx, stream);
  cor->compress_wrapup(out, outlen);
}

TESTFRAME::full_decompress(
    pszheader* header, Compressor* cor, u1* d_compressed, T* out,
    uninit_stream_t stream)
{
  auto in = d_compressed;
  auto d_space = out, d_xdata = out;

  cor->decompress_scatter(header, in, d_space, stream);
  cor->decompress_decode(header, in, stream);
  cor->decompress_predict(header, in, nullptr, d_xdata, stream);
  cor->decompress_collect_kerneltime();
}

TESTFRAME::pred_comp_decomp(
    pszctx* ctx, Compressor* cor, T* in, T* out, uninit_stream_t stream)
{
  auto header = new pszheader{};
  float time_sp;
  auto d_space = out, d_xdata = out;

  cor->compress_predict(ctx, in, stream);
  auto d_anchor = cor->mem->ac->dptr();

  psz::spv_scatter_naive<PROPER_GPU_BACKEND, T>(
      cor->mem->compact_val(), cor->mem->compact_idx(),
      cor->mem->compact_num_outliers(), d_space, &time_sp, stream);

  header->x = ctx->x, header->y = ctx->y, header->z = ctx->z;
  header->eb = ctx->eb, header->radius = ctx->radius;

  cor->decompress_predict(header, nullptr, d_anchor, d_xdata, stream);
}

TESTFRAME::pred_hist_comp(
    pszctx* ctx, Compressor* cor, T* in, uninit_stream_t stream,
    bool skip_print)
{
  float time_hist;
  auto len = ctx->data_len;
  auto booklen = ctx->radius * 2;

  cor->compress_predict(ctx, in, stream);

  /* In place of `cor->compress_histogram(ctx, in, stream);` */

  cor->mem->e->control({D2H});
  auto ectrl_gpu = cor->mem->e->dptr();
  auto ectrl_cpu = cor->mem->e->hptr();

  auto ht_gpu = new pszmem_cxx<u4>(booklen, 1, 1, "ht_gpu");
  ht_gpu->control({Malloc, MallocHost});

  auto ht_cpu = new pszmem_cxx<u4>(booklen, 1, 1, "ht_cpu");
  ht_cpu->control({MallocHost});

  psz::histsp<PROPER_GPU_BACKEND, u4>(
      ectrl_gpu, len, ht_gpu->dptr(), booklen, &time_hist, stream);
  psz::histsp<SEQ, u4>(
      ectrl_cpu, len, ht_cpu->hptr(), booklen, &time_hist, stream);

  ht_gpu->control({D2H});
  auto eq = std::equal(ht_gpu->hbegin(), ht_gpu->hend(), ht_cpu->hbegin());
  if (eq)
    printf("[psz::test] CPU and GPU hist result in the same.\n");
  else
    throw std::runtime_error(
        "[psz::test::error] CPU and GPU hist result differently.");

  if (not skip_print) {
    auto count = 0u;
    std::for_each(ht_gpu->hbegin(), ht_gpu->hend(), [&count](auto h) {
      if (h != 0) { printf("idx: %u\tfreq.: %u\n", count, h); }
      count++;
    });
  }
}

TESTFRAME::pred_hist_hf_comp(
    pszctx* ctx, Compressor* cor, T* in, uninit_stream_t stream)
{
  // TODO wrap up pred_hist_comp
}

#undef TESTFRAME

template class psz_testframe<f4>;
