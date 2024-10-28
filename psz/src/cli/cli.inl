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
#include "utils/config.hh"
#include "utils/err.hh"
#include "utils/query.hh"
#include "utils/viewer.hh"

namespace psz {

using namespace portable;

template <typename Data = float>
class CLI {
 private:
  using T = Data;

 public:
  CLI() = default;

  template <typename T>
  static void cli_dryrun(pszctx* ctx, bool dualquant = true)
  {
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    cudaStream_t stream;
    cudaStreamCreate(&stream);
#elif defined(PSZ_USE_1API)
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    dpct::queue_ptr stream = dev_ct1.create_queue();
#endif

    auto x = ctx->x, y = ctx->y, z = ctx->z;
    auto eb = ctx->eb;
    auto r2r = ctx->mode == Rel;
    auto fname = ctx->file_input;

    memobj<T>* original =
        new memobj<T>(x, y, z, "original", {MallocHost, Malloc});
    memobj<T>* reconst =
        new memobj<T>(x, y, z, "reconst", {MallocHost, Malloc});

    double max, min, rng;
    auto len = original->len();

    original->control({DBG});

    original->file(fname, FromFile)->control({ASYNC_H2D}, stream);
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
#elif defined(PSZ_USE_1API)
    stream->wait();
#endif

    if (r2r) original->extrema_scan(max, min, rng), eb *= rng;

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    psz::cuhip::GPU_lorenzo_dryrun(
        len, original->dptr(), reconst->dptr(), eb, stream);
#elif defined(PSZ_USE_1API)
    psz::dpcpp::GPU_lorenzo_dryrun(
        len, original->dptr(), reconst->dptr(), eb, stream);
#endif

    reconst->control({D2H});

    psz_summary stat;
    psz::utils::assess_quality<THRUST>(
        &stat, reconst->dptr(), original->dptr(), len);
    psz::utils::print_metrics_cross<T>(&stat, 0);

    // destroy
    original->control({FreeHost, Free});
    reconst->control({FreeHost, Free});

    delete original;
    delete reconst;

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    cudaStreamDestroy(stream);
#elif defined(PSZ_USE_1API)
    dev_ct1.destroy_queue(stream);
#endif
  }

 private:
  void write_compressed_to_disk(
      std::string compressed_name, psz_header* header, uint8_t* comped,
      size_t comp_len)
  {
    auto file = new memobj<uint8_t>(comp_len, "psz-archive", {MallocHost});

    file->dptr(comped)->control({D2H});
    memcpy(file->hptr(), header, sizeof(psz_header));  // put on-host header
    file->file(compressed_name.c_str(), ToFile);

    delete file;
  }

  void cli_compress(pszctx* const ctx, void* stream)
  {
    auto input = new memobj<T>(
        ctx->x, ctx->y, ctx->z, "uncomp'ed", {MallocHost, Malloc});

    input->file(ctx->file_input, FromFile)->control({H2D});

    uint8_t* comped;
    size_t comp_len;
    psz_header header;
    auto eb = ctx->eb;
    auto mode = ctx->mode;

    psz::TimeRecord timerecord;

    // the core of compression
    auto compressor = psz_create_from_context(ctx, get_len3(ctx));
    psz_compress(
        compressor, input->dptr(), get_len3(ctx), eb, mode, &comped, &comp_len,
        &header, (void*)&timerecord, stream);

    if (not ctx->there_is_memerr) {
      if (ctx->report_time) {
        printf(
            "\n\e[1m\e[31m"
            "REPORT::COMPRESSION::TIME"
            "\e[0m"
            "\n");
        psz_review_comp_time_breakdown(&timerecord, &header);
      }
      if (ctx->report_cr) {
        printf(
            "\n\e[1m\e[31m"
            "REPORT::COMPRESSION::FILE"
            "\e[0m"
            "\n");
        psz_review_comp_time_from_header(&header);
      }

      write_compressed_to_disk(
          std::string(ctx->file_input) + ".cusza", &header, comped, comp_len);
    }
    else {
      printf("\n*** exit on failure.\n");
    }

    psz_release(compressor);
    delete input;
  }

  void cli_decompress(pszctx* const ctx, void* stream)
  {
    // extract basename w/o suffix
    auto basename = std::string(ctx->file_input);
    basename = basename.substr(0, basename.rfind('.'));

    // all lengths in metadata
    auto compressed_len = psz_utils::filesize(ctx->file_input);

    auto comped =
        new memobj<uint8_t>(compressed_len, "comped", {MallocHost, Malloc});

    comped->file(ctx->file_input, FromFile)->control({H2D});

    auto header = (psz_header*)comped->hptr();
    auto len = pszheader_uncompressed_len(header);
    auto comp_len = pszheader_filesize(header);
    auto decomp_len = psz_len3{header->x, header->y, header->z};
    psz::TimeRecord timerecord;

    auto decomped = new memobj<T>(len, "decomp'ed", {MallocHost, Malloc});
    auto original = new memobj<T>(len, "original-cmp");

    // the core of decompression
    auto compressor = psz_create_from_header(header);
    psz_decompress(
        compressor, comped->dptr(), comp_len, decomped->dptr(), decomp_len,
        (void*)&timerecord, stream);

    if (ctx->report_time)
      psz_review_decompression(&timerecord, decomped->bytes());
    psz_review_decomp_time_from_header(header);
    psz::utils::view(header, decomped, original, ctx->file_compare);

    if (not ctx->skip_tofile)
      decomped->control({D2H})->file(
          std::string(basename + ".cuszx").c_str(), ToFile);

    psz_release(compressor);
    delete comped;
    delete decomped;
    delete original;
  }

 public:
  void dispatch(pszctx* ctx)
  {
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    cudaStream_t stream;
    CHECK_GPU(cudaStreamCreate(&stream));

    // TODO enable f8
    if (ctx->task_dryrun) cli_dryrun<float>(ctx);
    if (ctx->task_construct) cli_compress(ctx, stream);
    if (ctx->task_reconstruct) cli_decompress(ctx, stream);
    if (stream) cudaStreamDestroy(stream);

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
    if (ctx->task_dryrun) cli_dryrun<float>(ctx);
    if (ctx->task_construct) cli_compress(compressor, &q);
    if (ctx->task_reconstruct) cli_decompress(compressor, &q);

#endif
  }
};

}  // namespace psz

#endif
