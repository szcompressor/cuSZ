
/**
 * @file context.h
 * @author Jiannan Tian
 * @brief Argument parser (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on: 20-04-24
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory See LICENSE in top-level directory
 *
 */

#ifndef PSZ_CONTEXT_H
#define PSZ_CONTEXT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "cusz/type.h"
#include "header.h"

struct psz_cli_config {
  // filenames
  char opath[200];
  char file_input[500];
  char file_compare[500];

  // str for metadata
  char char_mode[4];
  char char_meta_eb[16];
  char char_predictor_name[sizeof("lorenzo-zigzag")];
  char char_hist_name[sizeof("histogram-centrality")];
  char char_codec1_name[sizeof("huffman-revisit")];
  char char_codec2_name[sizeof("huffman-revisit")];

  // dump intermediate
  bool dump_quantcode;
  bool dump_hist;
  bool dump_full_hf;

  bool task_construct;
  bool task_reconstruct;

  bool rel_range_scan;

  bool use_gpu_verify;

  bool skip_tofile;
  bool skip_hf;

  bool report_time;
  bool report_cr;
  bool verbose;
};
typedef psz_cli_config psz_cli_config;

struct psz_context {
  psz_header* header;
  psz_cli_config* cli;
  void* _future_comp_buf;
  void* compressor;
  void* compbuf;
  void* stream;

  psz_device device;
  uint16_t dict_size;
  size_t data_len;
  int ndim;
  psz_error_status last_error;
  // tracking error status
  bool there_is_memerr;
};

typedef struct psz_context psz_context;
typedef psz_context pszctx;
typedef psz_context psz_manager;
typedef psz_context psz_resource;
typedef psz_context psz_arguments;

void capi_psz_version();
void capi_psz_versioninfo();

// Return a pszctx instance with default values.
pszctx* pszctx_default_values();

// Modify an empty pszctx with default values.
void pszctx_set_default_values(pszctx*);

// Use a minimal workset as the return object.
pszctx* pszctx_minimal_workset(
    psz_dtype const dtype, psz_predtype const predictor, int const quantizer_radius,
    psz_codectype const codec);

void pszctx_set_rawlen(pszctx* ctx, size_t _x, size_t _y, size_t _z);
void pszctx_set_len(pszctx* ctx, psz_len3 len);
#define get_len3 pszctx_get_len3
psz_len3 pszctx_get_len3(pszctx* ctx);
void pszctx_create_from_argv(pszctx* ctx, int const argc, char** const argv);
void pszctx_create_from_string(pszctx* ctx, const char* in_str, bool dbg_print);

unsigned int CLI_x(psz_arguments* args);
unsigned int CLI_y(psz_arguments* args);
unsigned int CLI_z(psz_arguments* args);
unsigned int CLI_w(psz_arguments* args);
unsigned short CLI_radius(psz_arguments* args);
unsigned short CLI_bklen(psz_arguments* args);
psz_dtype CLI_dtype(psz_arguments* args);
psz_predtype CLI_predictor(psz_arguments* args);
psz_histotype CLI_hist(psz_arguments* args);
psz_codectype CLI_codec1(psz_arguments* args);
psz_codectype CLI_codec2(psz_arguments* args);
psz_mode CLI_mode(psz_arguments* args);
double CLI_eb(psz_arguments* args);

#ifdef __cplusplus
}
#endif

#endif /* PSZ_CONTEXT_H */
