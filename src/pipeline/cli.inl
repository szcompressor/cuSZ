/**
 * @file cli.cuh
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

#include <string>
#include <type_traits>

#include "context.h"
#include "cusz.h"
#include "cusz/type.h"
#include "dryrun.hh"
#include "framework.hh"
#include "header.h"
#include "mem/memseg_cxx.hh"
#include "utils/analyzer.hh"
#include "utils/cuda_err.cuh"
#include "utils/query.hh"
#include "utils/viewer.hh"

namespace cusz {

template <typename Data = float>
class CLI {
 private:
  using T = Data;

 public:
  CLI() = default;

  template <typename T>
  static void do_dryrun(cusz_context* ctx, bool dualquant = true)
  {
    Dryrunner<T> dryrun;

    uint3 xyz{ctx->x, ctx->y, ctx->z};
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    dryrun.init_dualquant_dryrun(xyz)
        .dualquant_dryrun(ctx->infile, ctx->eb, ctx->mode == Rel, stream)
        .destroy_dualquant_dryrun();

    cudaStreamDestroy(stream);
  }

 private:
  void write_compressed_to_disk(
      std::string compressed_name, uint8_t* compressed, size_t compressed_len)
  {
    auto file = new pszmem_cxx<uint8_t>(compressed_len, 1, 1, "cusza");
    file->dptr(compressed)
        ->control({MallocHost, D2H})
        ->file(compressed_name.c_str(), ToFile)
        ->control({FreeHost});

    delete file;
  }

  // template <typename compressor_t>
  void do_construct(
      cusz_context* ctx, cusz_compressor* compressor, cudaStream_t stream)
  {
    auto input = new pszmem_cxx<T>(ctx->x, ctx->y, ctx->z, "uncompressed");

    uint8_t* compressed;
    size_t compressed_len;
    cusz_header header;

    input->control({MallocHost, Malloc})
        ->file(ctx->infile, FromFile)
        ->control({H2D});

    // adjust eb
    if (ctx->mode == Rel) {
      double _1, _2, rng;
      input->extrema_scan(_1, _2, rng);
      ctx->eb *= rng;
    }

    TimeRecord timerecord;

    cusz_config* config = new cusz_config{.eb = ctx->eb, .mode = Rel};
    cusz_len uncomp_len = cusz_len{ctx->x, ctx->y, ctx->z, 1};

    cusz_compress(
        compressor, config, input->dptr(), uncomp_len, &compressed,
        &compressed_len, &header, (void*)&timerecord, stream);

    if (ctx->report_time)
      TimeRecordViewer::view_compression(
          &timerecord, input->m->bytes, compressed_len);
    write_compressed_to_disk(
        std::string(ctx->infile) + ".cusza", compressed, compressed_len);

    delete input;
  }

  // template <typename compressor_t>
  void do_reconstruct(
      cusz_context* ctx, cusz_compressor* compressor, cudaStream_t stream)
  {
    // extract basename w/o suffix
    auto basename = std::string(ctx->infile);
    basename = basename.substr(0, basename.rfind('.'));

    // all lengths in metadata
    auto compressed_len = psz_utils::filesize(ctx->infile);

    auto compressed =
        new pszmem_cxx<uint8_t>(compressed_len, 1, 1, "compressed");

    compressed->control({MallocHost, Malloc})
        ->file(ctx->infile, FromFile)
        ->control({H2D});

    auto header = new cusz_header;
    memcpy(header, compressed->hptr(), sizeof(cusz_header));
    auto len = psz_utils::uncompressed_len(header);

    auto decompressed = new pszmem_cxx<T>(len, 1, 1, "decompressed");
    decompressed->control({MallocHost, Malloc});

    auto original = new pszmem_cxx<T>(len, 1, 1, "original-cmp");

    TimeRecord timerecord;

    cusz_len decomp_len = cusz_len{header->x, header->y, header->z, 1};

    cusz_decompress(
        compressor, header, compressed->dptr(), psz_utils::filesize(header),
        decompressed->dptr(), decomp_len, (void*)&timerecord, stream);

    if (ctx->report_time)
      TimeRecordViewer::view_decompression(
          &timerecord, decompressed->m->bytes);
    psz::view(header, decompressed, original, ctx->original_file);

    if (not ctx->skip_tofile)
      decompressed->control({D2H})->file(
          std::string(basename + ".cuszx").c_str(), ToFile);

    decompressed->control({FreeHost, Free});
    delete decompressed;
    delete original;
  }

 public:
  // TODO determine dtype & predictor in here
  void dispatch(cusz_context* ctx)
  {
    // TODO disable predictor selection; to specify in another way
    // auto predictor = ctx->predictor;

    cusz_framework* framework = cusz_default_framework();
    cusz_compressor* compressor = cusz_create(framework, FP32);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // TODO enable f8
    if (ctx->task_dryrun) do_dryrun<float>(ctx);
    if (ctx->task_construct) do_construct(ctx, compressor, stream);
    if (ctx->task_reconstruct) do_reconstruct(ctx, compressor, stream);

    if (stream) cudaStreamDestroy(stream);
  }
};

}  // namespace cusz

#endif
