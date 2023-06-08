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
#include "cusz/type.h"
#include "dryrun.hh"
#include "framework.hh"
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

    if (not dualquant) {
      dryrun.init_dualquant_dryrun(xyz)
          .dualquant_dryrun(ctx->infile, ctx->eb, ctx->mode == Rel, stream)
          .destroy_dualquant_dryrun();
    }
    else {
      dryrun.init_generic_dryrun(xyz)
          .generic_dryrun(ctx->infile, ctx->eb, 512, ctx->mode == Rel, stream)
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
  void do_construct(
      cusz_context* ctx, cusz_compressor* compressor, cudaStream_t stream)
  {
    Capsule<T> input("uncompressed");
    uint8_t* compressed;
    size_t compressed_len;
    cusz_header header;
    auto len = ctx->data_len;
    // auto basename = ctx->infile;

    auto load_uncompressed = [&](std::string fname) {
      input
          .set_len(len)  //
          .mallochost()
          .malloc()
          .fromfile(fname)
          .host2device();
    };

    auto adjust_eb = [&]() {
      if (ctx->mode == Rel) ctx->eb *= input.prescan().get_rng();
    };

    /******************************************************************************/
    load_uncompressed(std::string(ctx->infile));
    adjust_eb();

    TimeRecord timerecord;

    cusz_config* config = new cusz_config{.eb = ctx->eb, .mode = Rel};
    cusz_len uncomp_len = cusz_len{ctx->x, ctx->y, ctx->z, 1};

    cusz_compress(
        compressor, config, input.dptr(), uncomp_len, &compressed,
        &compressed_len, &header, (void*)&timerecord, stream);

    if (ctx->report_time)
      TimeRecordViewer::view_compression(
          &timerecord, input.nbyte(), compressed_len);
    write_compressed_to_disk(
        std::string(ctx->infile) + ".cusza", compressed, compressed_len);
  }

  // template <typename compressor_t>
  void do_reconstruct(
      cusz_context* ctx, cusz_compressor* compressor, cudaStream_t stream)
  {
    Capsule<uint8_t> compressed("compressed");
    Capsule<T> decompressed("decompressed"), original("cmp");
    auto header = new cusz_header;

    auto basename = std::string(ctx->infile);
    // remove suffix ".cusza"
    basename = basename.substr(0, basename.rfind('.'));

    auto load_compressed = [&](std::string compressed_name) {
      auto compressed_len = psz_utils::get_filesize(compressed_name);
      compressed
          .set_len(compressed_len)  //
          .mallochost()
          .malloc()
          .fromfile(compressed_name)
          .host2device();
    };

    /******************************************************************************/

    load_compressed(basename + ".cusza");
    memcpy(header, compressed.hptr(), sizeof(cusz_header));
    auto len = psz_utils::get_uncompressed_len(header);

    decompressed  //
        .set_len(len)
        .mallochost()
        .malloc();
    original.set_len(len);

    TimeRecord timerecord;

    cusz_len decomp_len = cusz_len{header->x, header->y, header->z, 1};

    cusz_decompress(
        compressor, header, compressed.dptr(), psz_utils::get_filesize(header),
        decompressed.dptr(), decomp_len, (void*)&timerecord, stream);

    if (ctx->report_time)
      TimeRecordViewer::view_decompression(&timerecord, decompressed.nbyte());
    QualityViewer::view(header, decompressed, original, ctx->original_file);

    // try_write_decompressed_to_disk(decompressed, basename,
    // ctx->skip_tofile);

    if (not ctx->skip_tofile)
      decompressed.device2host().tofile(basename + ".cuszx");

    decompressed.freehost().free();
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
