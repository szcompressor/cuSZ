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

#include "cusz.h"
#include "dryrun.hh"
#include "framework.hh"
#include "utils/analyzer.hh"
#include "utils/cuda_err.cuh"
#include "utils/quality_viewer.hh"
#include "utils/query.hh"
#include "utils/timerecord_viewer.hh"

namespace cusz {

template <typename Data = float>
class CLI {
 private:
  using Header = cuszHEADER;
  using T = Data;

  using context_t = cuszCTX*;
  using header_t = cuszHEADER*;

 public:
  CLI() = default;

  template <typename T>
  static void cli_dryrun(context_t ctx, bool dualquant = true)
  {
    Dryrunner<T> dryrun;

    uint3 xyz{ctx->x, ctx->y, ctx->z};
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    if (not dualquant) {
      dryrun.init_dualquant_dryrun(xyz)
          .dualquant_dryrun(
              ctx->fname.fname, ctx->eb, ctx->mode == "r2r", stream)
          .destroy_dualquant_dryrun();
    }
    else {
      dryrun.init_generic_dryrun(xyz)
          .generic_dryrun(
              ctx->fname.fname, ctx->eb, 512, ctx->mode == "r2r", stream)
          .destroy_generic_dryrun();
    }
    cudaStreamDestroy(stream);
  }

 private:
  void write_compressed_to_disk(
      std::string compressed_name, uint8_t* compressed, size_t compressed_len)
  {
    Capsule<uint8_t> file("cusza");
    file.set_len(compressed_len)
        .set_dptr(compressed)
        .mallochost()
        .device2host()
        .tofile(compressed_name)
        .freehost()
        .free();
  }

  void try_write_decompressed_to_disk(
      Capsule<T>& xdata, std::string basename, bool skip_write)
  {
    if (not skip_write) xdata.device2host().tofile(basename + ".cuszx");
  }

  // template <typename compressor_t>
  void cli_construct(
      context_t ctx, cusz_compressor* compressor, cudaStream_t stream)
  {
    Capsule<T> input("uncompressed");
    uint8_t* compressed;
    size_t compressed_len;
    Header header;
    auto len = ctx->get_len();
    auto basename = ctx->fname.fname;

    auto load_uncompressed = [&](std::string fname) {
      input
          .set_len(len)  //
          .mallochost()
          .malloc()
          .fromfile(fname)
          .host2device();
    };

    auto adjust_eb = [&]() {
      if (ctx->mode == "r2r") ctx->eb *= input.prescan().get_rng();
    };

    /******************************************************************************/

    load_uncompressed(basename);
    adjust_eb();

    TimeRecord timerecord;

    cusz_config* config = new cusz_config{.eb = ctx->eb, .mode = Rel};
    cusz_len uncomp_len = cusz_len{ctx->x, ctx->y, ctx->z, 1};

    cusz_compress(
        compressor, config, input.dptr(), uncomp_len, &compressed,
        &compressed_len, &header, (void*)&timerecord, stream);

    if (ctx->report.time)
      TimeRecordViewer::view_compression(
          &timerecord, input.nbyte(), compressed_len);
    write_compressed_to_disk(basename + ".cusza", compressed, compressed_len);
  }

  // template <typename compressor_t>
  void cli_reconstruct(
      context_t ctx, cusz_compressor* compressor, cudaStream_t stream)
  {
    Capsule<uint8_t> compressed("compressed");
    Capsule<T> decompressed("decompressed"), original("cmp");
    auto header = new Header;
    auto basename = (*ctx).fname.fname;

    auto load_compressed = [&](std::string compressed_name) {
      auto compressed_len = ConfigHelper::get_filesize(compressed_name);
      compressed
          .set_len(compressed_len)  //
          .mallochost()
          .malloc()
          .fromfile(compressed_name)
          .host2device();
    };

    /******************************************************************************/

    load_compressed(basename + ".cusza");
    memcpy(header, compressed.hptr(), sizeof(Header));
    auto len = ConfigHelper::get_uncompressed_len(header);

    decompressed  //
        .set_len(len)
        .mallochost()
        .malloc();
    original.set_len(len);

    TimeRecord timerecord;

    cusz_len decomp_len = cusz_len{header->x, header->y, header->z, 1};

    cusz_decompress(
        compressor, header, compressed.dptr(),
        ConfigHelper::get_filesize(header), decompressed.dptr(), decomp_len,
        (void*)&timerecord, stream);

    if (ctx->report.time)
      TimeRecordViewer::view_decompression(&timerecord, decompressed.nbyte());
    QualityViewer::view(
        header, decompressed, original, (*ctx).fname.origin_cmp);
    try_write_decompressed_to_disk(
        decompressed, basename, (*ctx).skip.write2disk);

    decompressed.freehost().free();
  }

 public:
  // TODO determine dtype & predictor in here
  void dispatch(context_t ctx)
  {
    // TODO disable predictor selection; to specify in another way
    // auto predictor = (*ctx).predictor;

    cusz_framework* framework = cusz_default_framework();
    cusz_compressor* compressor = cusz_create(framework, FP32);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // TODO enable f8
    if ((*ctx).cli_task.dryrun) cli_dryrun<float>(ctx);
    if ((*ctx).cli_task.construct) cli_construct(ctx, compressor, stream);
    if ((*ctx).cli_task.reconstruct) cli_reconstruct(ctx, compressor, stream);

    if (stream) cudaStreamDestroy(stream);
  }
};

}  // namespace cusz

#endif
