/**
 * @file cli.cc
 * @brief CLI shell: pre-flight handling, subcommand dispatch, executor invocation.
 *
 * Owns no compute logic. Decides what to do (help / version / compress /
 * decompress) based on argv, then delegates: context.cc parses argv into ctx,
 * executor.cc runs the actual task.
 */

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

#include "cusz.h"
#include "executor.hh"
#include "query.hh"
#include "utils/format.hh"

using std::cerr;
using std::endl;

// Process exit status set. Keep `int` typed for compatibility with `exit()`.
namespace psz_exit {
constexpr int OK    = 0;  // success
constexpr int USAGE = 1;  // bad CLI args / validation failure
constexpr int DATA  = 2;  // corrupt or incompatible input file
constexpr int FATAL = 3;  // unexpected internal failure
}  // namespace psz_exit

static void psz_validate_cli_args(psz_ctx* ctx)
{
  bool abort = false;
  if (ctx->cli->file_input[0] == '\0') {
    cerr << LOG_ERR << "must specify input file" << endl;
    abort = true;
  }
  if (not ctx->cli->task_reduction and not ctx->cli->task_reconstruction) {
    cerr << LOG_ERR
         << "select `cusz compress` or `cusz decompress` (or use -z / -x without subcommand)."
         << endl;
    abort = true;
  }
  if (ctx->cli->task_reduction and ctx->header->dtype != F4 and ctx->header->dtype != F8) {
    cerr << LOG_ERR << "must specify data type (-t f32 or -t f64)" << endl;
    abort = true;
  }
  if (abort) { psz_print_document(false); exit(psz_exit::USAGE); }
}

int psz_run_from_CLI(int argc, char** argv)
{
  // Exit-only: subcommands and flags that never reach context setup.
  if (argc == 1) { psz_print_document(false); exit(psz_exit::OK); }
  {
    std::string sub(argv[1]);
    if (sub == "help")                              { psz_print_document(true); exit(psz_exit::OK); }
    if (sub == "version")                           { psz_version();            exit(psz_exit::OK); }
    if (sub == "versioninfo" or sub == "query-env") { psz_versioninfo();        exit(psz_exit::OK); }
  }
  for (int k = 1; k < argc; k++) {
    std::string a(argv[k]);
    if (a == "-h" or a == "--help")                              { psz_print_document(true); exit(psz_exit::OK); }
    if (a == "-v" or a == "--version")                           { psz_version();            exit(psz_exit::OK); }
    if (a == "-V" or a == "--versioninfo" or a == "--query-env") { psz_versioninfo();        exit(psz_exit::OK); }
  }

  auto args = pszctx_default_values();
  pszctx_create_from_argv(args, argc, argv);
  psz_validate_cli_args(args);

  if (args->cli->verbose) {
    CPU_QUERY;
    GPU_QUERY;
  }

  try {
    if (args->cli->task_reduction)      psz_compress_task(args);
    else                                psz_decompress_task(args);  // validation guarantees one is set
  }
  catch (const std::runtime_error& e) {
    cerr << LOG_ERR << e.what() << endl;
    exit(psz_exit::DATA);
  }
  catch (const std::exception& e) {
    cerr << LOG_ERR << "unexpected: " << e.what() << endl;
    exit(psz_exit::FATAL);
  }

  return psz_exit::OK;
}

int main(int argc, char** argv) { return psz_run_from_CLI(argc, argv); }
