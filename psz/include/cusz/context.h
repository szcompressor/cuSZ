#ifndef PSZ_CONTEXT_H
#define PSZ_CONTEXT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "cusz/header.h"
#include "cusz/type.h"

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
  void* buf;
  void* stream;

  psz_device device;
  uint16_t dict_size;
  size_t len_linear;
  int ndim;
  psz_error_status last_error;
  // tracking error status
  bool there_is_memerr;
};

typedef struct psz_context psz_context;
typedef psz_context psz_ctx;
typedef psz_context psz_manager;
typedef psz_context psz_resource;
typedef psz_context psz_args;

void psz_version();
void psz_versioninfo();

// Return a psz_ctx instance with default values.
psz_ctx* pszctx_default_values();

// Modify an empty psz_ctx with default values.
void pszctx_set_default_values(psz_ctx*);

// Use a minimal workset as the return object.
psz_ctx* pszctx_minimal_workset(
    psz_dtype const dtype, psz_predictor const predictor, int const quantizer_radius,
    psz_codec const codec);

void pszctx_set_rawlen(psz_ctx* ctx, size_t _x, size_t _y, size_t _z);
void pszctx_set_len(psz_ctx* ctx, psz_len3 len);
#define get_len3 pszctx_get_len3
psz_len3 pszctx_get_len3(psz_ctx* ctx);
void pszctx_create_from_argv(psz_ctx* ctx, int const argc, char** const argv);

unsigned int CLI_x(psz_args* args);
unsigned int CLI_y(psz_args* args);
unsigned int CLI_z(psz_args* args);
unsigned int CLI_w(psz_args* args);
unsigned short CLI_radius(psz_args* args);
unsigned short CLI_bklen(psz_args* args);
psz_dtype CLI_dtype(psz_args* args);
psz_predictor CLI_predictor(psz_args* args);
psz_hist CLI_hist(psz_args* args);
psz_codec CLI_codec1(psz_args* args);
psz_codec CLI_codec2(psz_args* args);
psz_mode CLI_mode(psz_args* args);
double CLI_eb(psz_args* args);
psz_interp_params* CLI_interp_params(psz_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif /* PSZ_CONTEXT_H */
