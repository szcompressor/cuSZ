// CLI-level task runner: compress/decompress workflows.
//
// Owns the cudaStream lifecycle, file I/O, dtype dispatch, and reporting.
// Reads configuration from psz_args via CLI_* accessors; calls into the
// library API (psz_compress_*, psz_decompress_*).

#include "executor.hh"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "compressor.hh"
#include "cusz.h"
#include "mem/cxx_backends.h"
#include "mem/gpu_stream.hh"
#include "utils/dtype_dispatch.hh"
#include "utils/io.hh"
#include "viewer.hh"

using _portable::utils::fromfile;
using _portable::utils::tofile;
using std::string;

// ---------------------------------------------------------------------------
// dtype-templated helpers (replace REPORT / COMPARE_WITH_ORIGIN / WRITE_TO_DISK macros)
// ---------------------------------------------------------------------------

template <typename T>
static void report_decomp(psz_args* args, psz_header* header, size_t len)
{
  if (args->cli->report_time) psz_review_decompression(nullptr, sizeof(T) * len);
  if (args->cli->verbose) psz_review_decomp_time_from_header(header);
}

template <typename T>
static void compare_with_origin(
    psz_args* args, cudaStream_t stream, T* d_decomped, size_t len, size_t comp_len,
    psz_header* header)
{
  if (string(args->cli->file_compare).empty()) return;
  sync_by_stream(stream);
  auto d_origin = MAKE_UNIQUE_DEVICE(T, len);
  auto h_origin = MAKE_UNIQUE_HOST(T, len);
  fromfile(args->cli->file_compare, h_origin.get(), len);
  memcpy_allkinds<H2D>(d_origin.get(), h_origin.get(), len);
  if (args->cli->verbose)
    psz::analysis::GPU_evaluate_quality_and_print(d_decomped, d_origin.get(), len, comp_len);
  else
    psz::analysis::GPU_evaluate_quality_and_print_concise(
        d_decomped, d_origin.get(), len, comp_len, header);
}

template <typename T>
static void write_decomp_to_disk(
    psz_args* args, cudaStream_t stream, T* d_decomped, size_t len, const string& basename)
{
  if (args->cli->skip_tofile) return;
  sync_by_stream(stream);
  auto h_decomped = MAKE_UNIQUE_HOST(T, len);
  memcpy_allkinds<D2H>(h_decomped.get(), d_decomped, len);
  tofile((basename + ".cuszx").c_str(), h_decomped.get(), sizeof(T) * len);
}

// ---------------------------------------------------------------------------
// Compression task
// ---------------------------------------------------------------------------

void psz_compress_task(psz_args* args)
{
  auto stream_owner = _portable::make_gpu_stream();
  cudaStream_t stream = stream_owner.get();

  auto len = CLI_x(args) * CLI_y(args) * CLI_z(args);

  uint8_t*      d_internal_compressed;
  psz_header    header;
  size_t        compressed_len;
  psz_resource* m{nullptr};

  _portable::utils::dtype_dispatch()
      .on<float, F4>([&](auto) {
        auto d_in = MAKE_UNIQUE_DEVICE(float, len);
        auto h_in = MAKE_UNIQUE_HOST(float, len);
        fromfile(args->cli->file_input, h_in.get(), len);
        memcpy_allkinds<H2D>(d_in.get(), h_in.get(), len);
        m = psz_create_resource_manager(
            F4, {CLI_x(args), CLI_y(args), CLI_z(args)},
            {CLI_predictor(args), CLI_hist(args), CLI_codec1(args), NULL_CODEC}, stream);
        m->cli                     = args->cli;
        m->header->pipeline.codec2 = CLI_codec2(args);
        psz_compress_float(
            m, {CLI_mode(args), CLI_eb(args), CLI_radius(args)}, d_in.get(), &header,
            &d_internal_compressed, &compressed_len);
      })
      .on<double, F8>([&](auto) {
        auto d_in = MAKE_UNIQUE_DEVICE(double, len);
        auto h_in = MAKE_UNIQUE_HOST(double, len);
        fromfile(args->cli->file_input, h_in.get(), len);
        memcpy_allkinds<H2D>(d_in.get(), h_in.get(), len);
        m = psz_create_resource_manager(
            F8, {CLI_x(args), CLI_y(args), CLI_z(args)},
            {CLI_predictor(args), CLI_hist(args), CLI_codec1(args), NULL_CODEC}, stream);
        m->cli                     = args->cli;
        m->header->pipeline.codec2 = CLI_codec2(args);
        psz_compress_double(
            m, {CLI_mode(args), CLI_eb(args), CLI_radius(args)}, d_in.get(), &header,
            &d_internal_compressed, &compressed_len);
      })
      .call(CLI_dtype(args));

  if (args->cli->report_time) fprintf(stderr, "Reporting time is disabled/to be updated.\n");

  if (args->cli->report_cr) {
    psz_review_comp_time_from_header(&header);
    if (args->cli->verbose) {
      printf("\n\e[1m\e[31mREPORT::COMPRESSION::FILE\e[0m\n");
      psz_review_comp_time_from_header_verbose(&header);
    }
  }

  if (not args->cli->skip_tofile) {
    auto compressed_name = string(args->cli->file_input) + ".cusza";
    auto file            = MAKE_UNIQUE_HOST(uint8_t, compressed_len);
    sync_by_stream(stream);
    memcpy_allkinds<D2H>(file.get(), d_internal_compressed, compressed_len);
    memcpy(file.get(), &header, sizeof(psz_header));  // put on-host header
    tofile(compressed_name.c_str(), file.get(), compressed_len);
  }

  // Sync before unique_ptr destructors run; see note in psz_decompress_task.
  sync_by_stream(stream);

  if (m) psz_release_resource(m);
}

// ---------------------------------------------------------------------------
// Header sanity check: rejects garbage inputs before they cause OOM/UB
// downstream. There is no magic number in psz_header (yet), so we cross-check:
//   1) on-disk file size matches the header's self-reported size
//   2) dtype is one of the supported values
// Catches the common cases: wrong file passed to decompress, truncated archive,
// version mismatch where struct layout has shifted.
// ---------------------------------------------------------------------------

static void check_header_or_throw(const psz_header* header, size_t on_disk_size)
{
  auto reported = pszheader_filesize(const_cast<psz_header*>(header));
  if (reported != on_disk_size)
    throw std::runtime_error(
        "input does not look like a .cusza archive: header reports " +
        std::to_string(reported) + " bytes but file is " +
        std::to_string(on_disk_size));
  if (header->dtype != F4 and header->dtype != F8)
    throw std::runtime_error(
        "input header dtype is invalid (" + std::to_string(header->dtype) +
        "); file may be corrupt or from an incompatible version");
}

// ---------------------------------------------------------------------------
// Decompression task
// ---------------------------------------------------------------------------

void psz_decompress_task(psz_args* args)
{
  auto         stream_owner = _portable::make_gpu_stream();
  cudaStream_t stream       = stream_owner.get();

  // extract basename w/o suffix
  auto basename = string(args->cli->file_input);
  basename      = basename.substr(0, basename.rfind('.'));

  // all lengths in metadata
  auto compressed_len = _portable::utils::filesize(args->cli->file_input);

  auto d_comped = MAKE_UNIQUE_DEVICE(uint8_t, compressed_len);
  auto h_comped = MAKE_UNIQUE_HOST(uint8_t, compressed_len);
  fromfile(args->cli->file_input, h_comped.get(), compressed_len);

  auto header = (psz_header*)h_comped.get();
  check_header_or_throw(header, compressed_len);

  memcpy_allkinds<H2D>(d_comped.get(), h_comped.get(), compressed_len);

  auto comp_len = pszheader_filesize(header);
  auto len      = pszheader_uncompressed_len(header);

  psz_resource* m = psz_create_resource_manager_from_header(header, stream);

  _portable::utils::dtype_dispatch()
      .on<float, F4>([&](auto) {
        auto d_decomped = MAKE_UNIQUE_DEVICE(float, len);
        psz_decompress_float(m, d_comped.get(), comp_len, d_decomped.get());
        report_decomp<float>(args, header, len);
        compare_with_origin<float>(args, stream, d_decomped.get(), len, comp_len, header);
        write_decomp_to_disk<float>(args, stream, d_decomped.get(), len, basename);
      })
      .on<double, F8>([&](auto) {
        auto d_decomped = MAKE_UNIQUE_DEVICE(double, len);
        psz_decompress_double(m, d_comped.get(), comp_len, d_decomped.get());
        report_decomp<double>(args, header, len);
        compare_with_origin<double>(args, stream, d_decomped.get(), len, comp_len, header);
        write_decomp_to_disk<double>(args, stream, d_decomped.get(), len, basename);
      })
      .call(CLI_dtype(m));

  // Sync before any unique_ptr destructors run, so we don't free GPU buffers
  // while the stream still has pending work. (cudaStreamDestroy syncs too,
  // but with RAII it runs LAST, after the buffers are gone.)
  sync_by_stream(stream);

  if (m) psz_release_resource(m);
}
