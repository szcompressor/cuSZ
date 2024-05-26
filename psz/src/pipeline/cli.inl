/**
 * @file cli.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-02-20
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CLI_CUH
#define CLI_CUH

#include "busyheader.hh"
#include "cusz.h"
#include "cusz/type.h"
#include "header.h"
#include "port.hh"
//
#include "context.h"
#include "dryrun.hh"
#include "mem.hh"
#include "stat/compare.hh"
#include "tehm.hh"
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
#include "utils/analyzer.hh"
#endif
#include "utils/err.hh"
#include "utils/query.hh"
#include "utils/viewer.hh"

namespace cusz {

using namespace portable;

template <typename Data = float>
class CLI {
 private:
  using T = Data;

 public:
  CLI() = default;

  template <typename T>
  static void do_dryrun(pszctx* ctx, bool dualquant = true)
  {
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    GpuStreamT stream;
    GpuStreamCreate(&stream);
#elif defined(PSZ_USE_1API)
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    dpct::queue_ptr stream = dev_ct1.create_queue();
#endif

    auto x = ctx->x, y = ctx->y, z = ctx->z;
    auto eb = ctx->eb;
    auto r2r = ctx->mode == Rel;
    auto fname = ctx->file_input;

    pszmem_cxx<T>* original =
        new pszmem_cxx<T>(x, y, z, "original", {MallocHost, Malloc});
    pszmem_cxx<T>* reconst =
        new pszmem_cxx<T>(x, y, z, "reconst", {MallocHost, Malloc});

    double max, min, rng;
    auto len = original->len();

    original->control({DBG});

    original->file(fname, FromFile)->control({ASYNC_H2D}, stream);
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    CHECK_GPU(GpuStreamSync((GpuStreamT)stream));
#elif defined(PSZ_USE_1API)
    stream->wait();
#endif

    if (r2r) original->extrema_scan(max, min, rng), eb *= rng;

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    psz::cu_hip::dryrun(len, original->dptr(), reconst->dptr(), eb, stream);
#elif defined(PSZ_USE_1API)
    psz::dpcpp::dryrun(len, original->dptr(), reconst->dptr(), eb, stream);
#endif

    reconst->control({D2H});

    psz_summary stat;
    psz::assess_quality<SEQ>(&stat, reconst->hptr(), original->hptr(), len);
    psz::print_metrics_cross<T>(&stat, 0, true);

    // destroy
    original->control({FreeHost, Free});
    reconst->control({FreeHost, Free});

    delete original;
    delete reconst;

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    GpuStreamDestroy(stream);
#elif defined(PSZ_USE_1API)
    dev_ct1.destroy_queue(stream);
#endif
  }

 private:
  void write_compressed_to_disk(
      std::string compressed_name, pszheader* header, uint8_t* compressed,
      size_t compressed_len)
  {
    auto file = new pszmem_cxx<uint8_t>(compressed_len, "cusza", {MallocHost});

    file->dptr(compressed)->control({D2H});
    memcpy(file->hptr(), header, sizeof(pszheader));  // put on-host header
    file->file(compressed_name.c_str(), ToFile);

    delete file;
  }

  // template <typename compressor_t>
  void do_construct(pszctx* ctx, psz_compressor* compressor, void* stream)
  {
    auto input = new pszmem_cxx<T>(
        ctx->x, ctx->y, ctx->z, "uncompressed", {MallocHost, Malloc});

    uint8_t* compressed;
    size_t compressed_len;
    pszheader header;

    input->file(ctx->file_input, FromFile)->control({H2D});

    // adjust eb
    if (ctx->mode == Rel) {
      double _1, _2, rng;
      input->extrema_scan(_1, _2, rng);
      ctx->eb *= rng;
    }

    psz::TimeRecord timerecord;

    pszlen uncomp_len = pszlen{ctx->x, ctx->y, ctx->z, 1};

    psz_compress_init(compressor, uncomp_len, ctx);

    psz_compress(
        compressor, input->dptr(), uncomp_len, &compressed, &compressed_len,
        &header, (void*)&timerecord, stream);

    if (not ctx->there_is_memerr) {
      printf("\n(c) COMPRESSION REPORT\n");

      if (ctx->report_time)
        psz::TimeRecordViewer::view_timerecord(&timerecord, &header);
      if (ctx->report_cr) psz::TimeRecordViewer::view_cr(&header);

      write_compressed_to_disk(
          std::string(ctx->file_input) + ".cusza", &header, compressed,
          compressed_len);
    }
    else {
      printf("\n*** exit on failure.\n");
    }

    // delete data buffers external to compressor
    delete input;
  }

  // template <typename compressor_t>
  void do_reconstruct(pszctx* ctx, psz_compressor* compressor, void* stream)
  {
    // extract basename w/o suffix
    auto basename = std::string(ctx->file_input);
    basename = basename.substr(0, basename.rfind('.'));

    // all lengths in metadata
    auto compressed_len = psz_utils::filesize(ctx->file_input);

    auto compressed = new pszmem_cxx<uint8_t>(
        compressed_len, "compressed", {MallocHost, Malloc});

    compressed->file(ctx->file_input, FromFile)->control({H2D});

    auto header = new psz_header;
    memcpy(header, compressed->hptr(), sizeof(psz_header));
    auto len = psz_utils::uncompressed_len(header);

    auto decompressed =
        new pszmem_cxx<T>(len, "decompressed", {MallocHost, Malloc});
    auto original = new pszmem_cxx<T>(len, "original-cmp");

    psz::TimeRecord timerecord;

    pszlen decomp_len = pszlen{header->x, header->y, header->z, 1};

    psz_decompress_init(compressor, header);
    psz_decompress(
        compressor, compressed->dptr(), psz_utils::filesize(header),
        decompressed->dptr(), decomp_len, (void*)&timerecord, stream);

    if (ctx->report_time)
      psz::TimeRecordViewer::view_decompression(
          &timerecord, decompressed->bytes());
    psz::view(header, decompressed, original, ctx->file_compare);

    if (not ctx->skip_tofile)
      decompressed->control({D2H})->file(
          std::string(basename + ".cuszx").c_str(), ToFile);

    // delete data buffers external to compressor
    delete compressed;
    delete decompressed;
    delete original;
  }

 public:
  // TODO determine dtype & predictor in here
  void dispatch(pszctx* ctx)
  {
    // TODO disable predictor selection; to specify in another way
    // auto predictor = ctx->predictor;

    // TODO make it a value rather than a pointer
    psz_framework* framework = pszdefault_framework();
    psz_compressor* compressor = psz_create(framework, F4);

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    GpuStreamT stream;
    CHECK_GPU(GpuStreamCreate(&stream));

    // TODO enable f8
    if (ctx->task_dryrun) do_dryrun<float>(ctx);
    if (ctx->task_construct) do_construct(ctx, compressor, stream);
    if (ctx->task_reconstruct) do_reconstruct(ctx, compressor, stream);
    if (stream) GpuStreamDestroy(stream);

#elif defined(PSZ_USE_1API)

    sycl::queue q;
    auto plist = sycl::property_list(
        sycl::property::queue::in_order(),
        sycl::property::queue::enable_profiling());

    if (ctx->device == CPU)
      q = sycl::queue(sycl::cpu_selector_v, plist);
    else if (ctx->device == INTELGPU)
      q = sycl::queue(sycl::gpu_selector_v, plist);
    else
      q = sycl::queue(sycl::default_selector_v, plist);

    // TODO enable f8
    if (ctx->task_dryrun) do_dryrun<float>(ctx);
    if (ctx->task_construct) do_construct(ctx, compressor, &q);
    if (ctx->task_reconstruct) do_reconstruct(ctx, compressor, &q);

#endif

    // TODO mirrored with creation
    delete framework;

    psz_release(compressor);
    // delete compressor;
  }
};

}  // namespace cusz

#endif
