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

#include "cusz.h"
#include "tehm.hh"
// #include "utils/config.hh"
#include <fstream>

#include "utils/err.hh"
#include "utils/query.hh"
#include "utils/viewer.hh"

namespace psz {

template <typename T>
using memobj = _portable::memobj<T>;

template <typename T>
class CLI {
 private:
  void write_compressed_to_disk(
      std::string compressed_name, psz_header* header, uint8_t* comped, size_t comp_len);

  void cli_compress(pszctx* const ctx, void* stream);
  void cli_decompress(pszctx* const ctx, void* stream);
  [[deprecated]] static void cli_dryrun();

 public:
  CLI() = default;

  void dispatch(pszctx* ctx);
};

template <typename T>
void CLI<T>::cli_dryrun()
{
  printf("[psz::warn] CLI::dryun is disabled.");
  exit(0);
}

template <typename T>
void CLI<T>::cli_compress(pszctx* const ctx, void* stream)
{
  auto write_compressed_to_disk = [](std::string compressed_name, psz_header* header,
                                     uint8_t* comped, size_t comp_len) {
    auto file = new memobj<uint8_t>(comp_len, "psz-archive", {MallocHost});

    file->dptr(comped)->control({D2H});
    memcpy(file->hptr(), header, sizeof(psz_header));  // put on-host header
    file->file(compressed_name.c_str(), ToFile);

    delete file;
  };

  auto input = new memobj<T>(ctx->x, ctx->y, ctx->z, "uncomp'ed", {MallocHost, Malloc});

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
      compressor, input->dptr(), get_len3(ctx), eb, mode, &comped, &comp_len, &header,
      (void*)&timerecord, stream);

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

    if (not ctx->skip_tofile)
      write_compressed_to_disk(std::string(ctx->file_input) + ".cusza", &header, comped, comp_len);
  }
  else {
    printf("\n*** exit on failure.\n");
  }

  psz_release(compressor);
  delete input;
}

template <typename T>
void CLI<T>::cli_decompress(pszctx* const ctx, void* stream)
{
  // extract basename w/o suffix
  auto basename = std::string(ctx->file_input);
  basename = basename.substr(0, basename.rfind('.'));

  // all lengths in metadata
  auto filesize = [](std::string fname) -> size_t {
    std::ifstream in(fname.c_str(), std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
  };

  auto compressed_len = filesize(ctx->file_input);

  auto comped = new memobj<uint8_t>(compressed_len, "comped", {MallocHost, Malloc});

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
      compressor, comped->dptr(), comp_len, decomped->dptr(), decomp_len, (void*)&timerecord,
      stream);

  if (ctx->report_time) psz_review_decompression(&timerecord, decomped->bytes());
  psz_review_decomp_time_from_header(header);
  psz::analysis::view(header, decomped, original, ctx->file_compare);

  if (not ctx->skip_tofile)
    decomped->control({D2H})->file(std::string(basename + ".cuszx").c_str(), ToFile);

  psz_release(compressor);
  delete comped;
  delete decomped;
  delete original;
}

template <typename T>
void CLI<T>::dispatch(pszctx* ctx)
{
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
  cudaStream_t stream;
  CHECK_GPU(cudaStreamCreate(&stream));

  // TODO enable f8
  if (ctx->task_dryrun) cli_dryrun();
  if (ctx->task_construct) cli_compress(ctx, stream);
  if (ctx->task_reconstruct) cli_decompress(ctx, stream);
  if (stream) cudaStreamDestroy(stream);

#elif defined(PSZ_USE_1API)

  sycl::queue q;
  auto plist = sycl::property_list(
      sycl::property::queue::in_order(), sycl::property::queue::enable_profiling());

  if (ctx->device == CPU)
    q = sycl::queue(sycl::cpu_selector_v, plist);
  else if (ctx->device == INTELGPU)
    q = sycl::queue(sycl::gpu_selector_v, plist);
  else
    q = sycl::queue(sycl::default_selector_v, plist);

  if (ctx->task_dryrun) cli_dryrun();
  if (ctx->task_construct) cli_compress(compressor, &q);
  if (ctx->task_reconstruct) cli_decompress(compressor, &q);

#endif
}

}  // namespace psz

int main(int argc, char** argv)
{
  auto ctx = pszctx_default_values();
  pszctx_create_from_argv(ctx, argc, argv);

  if (ctx->verbose) {
    CPU_QUERY;
    GPU_QUERY;
  }

  psz::CLI<float> psz_cli;
  psz_cli.dispatch(ctx);

  delete ctx;

  return 0;
}