
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

#ifndef A93D242E_0C7C_44BD_AE43_B3A26084971A
#define A93D242E_0C7C_44BD_AE43_B3A26084971A

#ifdef __cplusplus
extern "C" {
#endif

#include "cusz/type.h"
#include "stdint.h"

struct psz_context {
  pszdevice device;

  psz_predtype pred_type;

  // sizes
  uint32_t x{1}, y{1}, z{1}, w{1};
  size_t data_len{1};
  size_t splen{0};
  int ndim{-1};

  // filenames
  char demodata_name[40];
  char opath[200];
  char file_input[500];
  char file_compare[500];
  char file_prebuilt_hist_top1[500];
  char file_prebuilt_hfbk[500];

  // pipeline config
  psz_dtype dtype{F4};
  psz_mode mode{Rel};
  double eb{0.0};
  int dict_size{1024}, radius{512};
  int prebuilt_bklen{1024}, prebuilt_nbk{1000};

  // spv gather-scatter config, tmp. unused
  float nz_density{0.2};
  float nz_density_factor{5};

  // codec config
  //   uint32_t codecs_in_use{0b01};
  //   int quant_bytewidth{2}, huff_bytewidth{4};
  int vle_sublen{512}, vle_pardeg{-1};

  // ???
  char dbgstr_pred[10];

  // dump intermediate
  bool dump_quantcode{false};
  bool dump_hist{false};

  bool task_construct{false};
  bool task_reconstruct{false};
  bool task_dryrun{false};
  bool task_experiment{false};

  bool prep_binning{false};
  //   bool prep_logtransform{false};
  bool prep_prescan{false};

  bool use_demodata{false};
  bool use_autotune_phf{true};
  bool use_gpu_verify{false};
  bool use_prebuilt_hfbk{false};

  bool skip_tofile{false};
  bool skip_hf{false};

  bool report_time{false};
  bool report_cr{false};
  bool report_cr_est{false};
  bool verbose{false};

  // tracking error status
  bool there_is_memerr{false};
};

typedef struct psz_context psz_context;
typedef psz_context pszctx;

void pszctx_print_document(bool full_document);
void pszctx_parse_argv(pszctx* ctx, int const argc, char** const argv);
void pszctx_parse_length(pszctx* ctx, const char* lenstr);
void pszctx_parse_length_zyx(pszctx* ctx, const char* lenstr);
void pszctx_parse_control_string(
    pszctx* ctx, const char* in_str, bool dbg_print);
void pszctx_validate(pszctx* ctx);
void pszctx_load_demo_datasize(pszctx* ctx, void* demodata_name);
void pszctx_set_rawlen(
    pszctx* ctx, size_t _x, size_t _y, size_t _z, size_t _w);
void pszctx_set_len(pszctx* ctx, pszlen len);
void pszctx_set_report(pszctx* ctx, const char* in_str);
void pszctx_set_dumping(pszctx* ctx, const char* in_str);
void pszctx_set_radius(pszctx* ctx, int _);
// void pszctx_set_huffbyte(pszctx* ctx, int _);
void pszctx_set_huffchunk(pszctx* ctx, int _);
void pszctx_set_densityfactor(pszctx* ctx, int _);
void pszctx_create_from_argv(pszctx* ctx, int const argc, char** const argv);
void pszctx_create_from_string(
    pszctx* ctx, const char* in_str, bool dbg_print);

#ifdef __cplusplus
}
#endif

#endif /* A93D242E_0C7C_44BD_AE43_B3A26084971A */
