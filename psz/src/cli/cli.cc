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

#include <fstream>

#include "api_v2.h"
#include "compressor.hh"
#include "cusz.h"
#include "mem/cxx_backends.h"
#include "utils/err.hh"
#include "utils/io.hh"
#include "utils/query.hh"
#include "utils/viewer.hh"

using _portable::utils::fromfile;
using _portable::utils::tofile;

#define REPORT(T)                                                                     \
  if (args->cli->report_time) psz_review_decompression(&timerecord, sizeof(T) * len); \
  psz_review_decomp_time_from_header(header);

#define COMPARE_WITH_ORIGIN(T)                                 \
  if (string(args->cli->file_compare) != "") {                 \
    auto d_origin = MAKE_UNIQUE_DEVICE(T, len);                \
    auto h_origin = MAKE_UNIQUE_HOST(T, len);                  \
    fromfile(args->cli->file_compare, h_origin.get(), len);    \
    memcpy_allkinds<H2D>(d_origin.get(), h_origin.get(), len); \
    psz::analysis::GPU_evaluate_quality_and_print(             \
        d_decomped.get(), d_origin.get(), len, comp_len);      \
  }

#define WRITE_TO_DISK(T)                                                                 \
  if (not args->cli->skip_tofile) {                                                      \
    auto h_decomped = MAKE_UNIQUE_HOST(T, len);                                          \
    memcpy_allkinds<D2H>(h_decomped.get(), d_decomped.get(), len);                       \
    tofile(std::string(basename + ".cuszx").c_str(), h_decomped.get(), sizeof(T) * len); \
  }

int psz_run_from_CLI(int argc, char** argv)
{
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto args = pszctx_default_values();
  pszctx_create_from_argv(args, argc, argv);

  if (args->cli->verbose) {
    CPU_QUERY;
    GPU_QUERY;
  }

  if (args->cli->task_construct) {
    auto len = CLI_x(args) * CLI_y(args) * CLI_z(args);

    uint8_t* d_internal_compressed;
    psz_header header;
    size_t compressed_len;
    psz::TimeRecord timerecord;

    psz_resource* m{nullptr};

    if (CLI_dtype(args) == F4) {
      auto d_in = MAKE_UNIQUE_DEVICE(float, len);
      auto h_in = MAKE_UNIQUE_HOST(float, len);
      fromfile(args->cli->file_input, h_in.get(), len);
      memcpy_allkinds<H2D>(d_in.get(), h_in.get(), len);

      m = psz_create_resource_manager(F4, CLI_x(args), CLI_y(args), CLI_z(args), stream);
      psz_compress_float(
          m,
          {CLI_predictor(args), CLI_hist(args), CLI_codec1(args), NULL_CODEC, CLI_mode(args),
           CLI_eb(args), CLI_radius(args)},
          d_in.get(), &header, &d_internal_compressed, &compressed_len);
    }
    else if (CLI_dtype(args) == F8) {
      auto d_in = MAKE_UNIQUE_DEVICE(double, len);
      auto h_in = MAKE_UNIQUE_HOST(double, len);
      fromfile(args->cli->file_input, h_in.get(), len);
      memcpy_allkinds<H2D>(d_in.get(), h_in.get(), len);

      m = psz_create_resource_manager(F8, CLI_x(args), CLI_y(args), CLI_z(args), stream);
      psz_compress_double(
          m,
          {CLI_predictor(m), CLI_hist(args), CLI_codec1(args), NULL_CODEC, CLI_mode(args),
           CLI_eb(args), CLI_radius(args)},
          d_in.get(), &header, &d_internal_compressed, &compressed_len);
    }

    if (args->cli->report_time) {
      printf("\n\e[1m\e[31mREPORT::COMPRESSION::TIME\e[0m\n");
      // psz_review_comp_time_breakdown(&timerecord, &header);
    }
    if (args->cli->report_cr) {
      printf("\n\e[1m\e[31mREPORT::COMPRESSION::FILE\e[0m\n");
      psz_review_comp_time_from_header(&header);
    }

    if (not args->cli->skip_tofile) {
      auto compressed_name = std::string(args->cli->file_input) + ".cusza";
      auto file = MAKE_UNIQUE_HOST(uint8_t, compressed_len);
      memcpy_allkinds<D2H>(file.get(), d_internal_compressed, compressed_len);
      memcpy(file.get(), &header, sizeof(psz_header));  // put on-host header
      tofile(compressed_name.c_str(), file.get(), compressed_len);
    }

    if (m) psz_release_resource(m);
  }
  else if (args->cli->task_reconstruct) {
    // extract basename w/o suffix
    auto basename = std::string(args->cli->file_input);
    basename = basename.substr(0, basename.rfind('.'));

    // all lengths in metadata
    auto filesize = [](std::string fname) -> size_t {
      std::ifstream in(fname.c_str(), std::ifstream::ate | std::ifstream::binary);
      return in.tellg();
    };

    auto compressed_len = filesize(args->cli->file_input);

    auto d_comped = MAKE_UNIQUE_DEVICE(uint8_t, compressed_len);
    auto h_comped = MAKE_UNIQUE_HOST(uint8_t, compressed_len);
    fromfile(args->cli->file_input, h_comped.get(), compressed_len);
    memcpy_allkinds<H2D>(d_comped.get(), h_comped.get(), compressed_len);

    auto header = (psz_header*)h_comped.get();
    auto comp_len = pszheader_filesize(header);
    auto len = pszheader_uncompressed_len(header);
    psz::TimeRecord timerecord;

    psz_resource* m = psz_create_resource_manager_from_header(header, stream);

    if (CLI_dtype(m) == F4) {
      auto d_decomped = MAKE_UNIQUE_DEVICE(float, len);
      psz_decompress_float(m, d_comped.get(), comp_len, d_decomped.get());
      REPORT(float);
      COMPARE_WITH_ORIGIN(float);
      WRITE_TO_DISK(float);
    }
    else if (CLI_dtype(m) == F8) {
      auto d_decomped = MAKE_UNIQUE_DEVICE(double, len);
      psz_decompress_double(m, d_comped.get(), comp_len, d_decomped.get());
      REPORT(double);
      COMPARE_WITH_ORIGIN(double);
      WRITE_TO_DISK(double);
    }

    if (m) psz_release_resource(m);
  }
  else {
    cerr << "No task specified." << endl;
    exit(1);
  }

  cudaStreamDestroy(stream);

  return 0;
}

int main(int argc, char** argv)
{
  psz_run_from_CLI(argc, argv);
  return 0;
}