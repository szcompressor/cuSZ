#ifndef _PSZ_CLI_EXECUTOR_HH
#define _PSZ_CLI_EXECUTOR_HH

#include "cusz/context.h"

// Run the compression task: read file_input, compress according to ctx,
// optionally report and write `<file_input>.cusza` to disk.
void psz_compress_task(psz_args* args);

// Run the decompression task: read `<file_input>.cusza`, decompress,
// optionally compare against `file_compare` and write `<basename>.cuszx` to disk.
void psz_decompress_task(psz_args* args);

#endif
