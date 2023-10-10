/**
 * @file bin_pipeline.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-06
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include <stdexcept>

#include "busyheader.hh"
#include "context.h"
#include "cusz.h"
#include "ex_utils.hh"
#include "hf/hf.hh"
#include "kernel.hh"
#include "mem.hh"
#include "port.hh"
#include "stat.hh"
#include "tehm.hh"
// #include "utils.hh"
#include "utils/viewer.hh"

using Compressor = cusz::Compressor<cusz::TEHM<f4>>;
using BYTE = u1;

#define PRINT_STATUS                                              \
  {                                                               \
    printf("\033[1mfname=\033[0m\"%s\"\n", ifn);                  \
    printf("\033[1m(x, y, z)=\033[0m(%lu, %lu, %lu)\n", x, y, z); \
    printf(                                                       \
        "\033[1m(eb, mode, radius)=\033[0m(%lf, %s, %d)\n", eb,   \
        mode == REL ? "REL" : "ABS", radius);                     \
    printf(                                                       \
        "\033[1mpredictor=\033[0m%s\n",                           \
        Predictor == SPLINE3 ? "spline3" : "lorenzo");            \
    if (mode == REL) printf("\033[1meb_rel=\033[0m%lf\n", eb);    \
  }

string dtype, etype, mode, pred, eb_str;
f8 _1, _2, rng;

namespace {

szt tune_coarse_huffman_sublen(szt len)
{
  int current_dev = 0;
  GpuSetDevice(current_dev);
  GpuDeviceProp dev_prop{};
  GpuGetDeviceProperties(&dev_prop, current_dev);

  // auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  auto nSM = dev_prop.multiProcessorCount;
  auto allowed_block_dim = dev_prop.maxThreadsPerBlock;
  auto deflate_nthread =
      allowed_block_dim * nSM / HuffmanHelper::DEFLATE_CONSTANT;
  auto optimal_sublen = psz_utils::get_npart(len, deflate_nthread);
  optimal_sublen =
      psz_utils::get_npart(optimal_sublen, HuffmanHelper::BLOCK_DIM_DEFLATE) *
      HuffmanHelper::BLOCK_DIM_DEFLATE;

  return optimal_sublen;
}

int ndim(dim3 l)
{
  auto n = 3;
  if (l.z == 1) n = 2;
  if (l.y == 1) n = 1;
  return n;
}

void radius_legal(int const radius, int const sizeof_T)
{
  size_t upper_bound = 1lu << (sizeof_T * 8);
  // cout << upper_bound << endl;
  // cout << radius * 2 << endl;
  if ((radius * 2) > upper_bound)
    throw std::runtime_error("Radius overflows error-quantization type.");
}

string suffix(bool compat = false)
{
  if (not compat)
    return pred + "_" + string(dtype == "f" ? "f4" : "f8") + etype + "_" +
           mode + eb_str;
  else
    return pred + "_" + string(dtype == "f" ? "f4" : "f8") + "u4" + "_" +
           mode + eb_str;
}

}  // namespace

template <typename T = f4>
float print_GBps(szt len, float time_in_ms, string fn_name)
{
  auto B_to_GiB = 1.0 * 1024 * 1024 * 1024;
  auto GiBps = len * sizeof(T) * 1.0 / B_to_GiB / (time_in_ms / 1000);
  auto title = "[psz::info::res::" + fn_name + "]";
  printf(
      "%s shortest time (ms): %.6f\thighest throughput (GiB/s): %.2f\n",
      title.c_str(), time_in_ms, GiBps);
  return GiBps;
}

template <typename T = f4>
void full_compress(
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

template <typename T = f4>
void full_decompress(
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

template <typename T = f4>
void pred_comp_decomp(
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

template <typename T = f4>
void pred_hist_comp(
    pszctx* ctx, Compressor* cor, T* in, uninit_stream_t stream,
    bool skip_print = false)
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

template <typename T = f4>
void pred_hist_hf_comp(
    pszctx* ctx, Compressor* cor, T* in, uninit_stream_t stream)
{
  // TODO wrap up pred_hist_comp
}

template <typename T = f4>
void run(pszctx* ctx, string const subcmd, char* fname, char* config_str)
{
  auto header = new pszheader{};
  szt outlen;

  pszctx_parse_control_string(ctx, config_str, true);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto data = new pszmem_cxx<f4>(ctx->x, ctx->y, ctx->z, "uncompressed");
  data->control({MallocHost, Malloc})->file(fname, FromFile)->control({H2D});

  auto xdata = new pszmem_cxx<f4>(ctx->x, ctx->y, ctx->z, "decompressed");
  xdata->control({MallocHost, Malloc});

  auto cmp = new pszmem_cxx<f4>(ctx->x, ctx->y, ctx->z, "cmp");

  // adjust eb
  if (ctx->mode == Rel) {
    double _1, _2, rng;
    data->extrema_scan(_1, _2, rng);
    ctx->eb *= rng;
  }

  BYTE* d_compressed{nullptr};

  cusz::CompressorHelper::autotune_coarse_parhf(ctx);

  auto cor = new Compressor();
  cor->init(ctx);

  if (subcmd == "full") {
    full_compress(ctx, cor, data->dptr(), &d_compressed, &outlen, stream);
    cor->export_header(header);
    psz::TimeRecordViewer::view_cr(header);
    full_decompress(header, cor, d_compressed, xdata->dptr(), stream);
    psz::view(header, xdata, cmp, fname);
  }
  else if (subcmd == "pred-only") {
    pred_comp_decomp(ctx, cor, data->dptr(), xdata->dptr(), stream);
    psz::eval_dataquality_gpu(
        xdata->dptr(), data->dptr(), data->len(), data->bytes());
  }
  else if (subcmd == "pred-hist") {
    pred_hist_comp(ctx, cor, data->dptr(), stream);
  }

  delete data;
  delete xdata;
  delete cor;

  cudaStreamDestroy(stream);
}

int main(int argc, char** argv)
{
  //// help
  if (argc < 4) {
    printf("0     1       2      3       [4]\n");
    printf("PROG  subcmd  fname  config  [if dump related]\n");
    printf("subcmd contains:\n");
    printf("  full\n");
    printf("  pred-only\n");
    printf("\n");
    printf("k-v config contains:\n");
    printf("  type={f4}\n");
    printf("  mode={abs,r2r}\n");
    printf("  eb=[number]\n");
    printf("  len/dim3=[X]x[Y]x[Z] or size=[Z]x[Y]x[X]\n");
    printf("  radius=[number]\n");
    printf("  predictor={lorenzo,spline}\n");
    exit(0);
  }

  auto ctx = new pszctx{};

  //// read argv
  auto subcmd = argv[1];
  auto fname = argv[2];
  auto config_str = argv[3];

  run<f4>(ctx, string(subcmd), fname, config_str);

  return 0;
}
