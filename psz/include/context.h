
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
  psz_device device;

  psz_dtype dtype;
  psz_predtype pred_type;
  psz_histogramtype hist_type;
  psz_codectype codec1_type;

  // pipeline config
  psz_mode mode;
  double eb;
  int dict_size, radius;
  int prebuilt_bklen, prebuilt_nbk;

  // spv gather-scatter config, tmp. unused
  float nz_density;
  float nz_density_factor;

  // codec config
  int vle_sublen, vle_pardeg;

  // sizes
  uint32_t x, y, z, w;
  size_t data_len;
  size_t splen;
  int ndim;
  psz_error_status last_error;

  // internal logging
  double user_input_eb, logging_min, logging_max;

  // filenames
  char demodata_name[40];
  char opath[200];
  char file_input[500];
  char file_compare[500];
  char file_prebuilt_hist_top1[500];
  char file_prebuilt_hfbk[500];

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
  bool task_dryrun;
  bool task_experiment;

  bool prep_binning;
  //   bool prep_logtransform;
  bool prep_prescan;

  bool use_demodata;
  bool use_autotune_phf;
  bool use_gpu_verify;
  bool use_prebuilt_hfbk;

  bool skip_tofile;
  bool skip_hf;

  bool report_time;
  bool report_cr;
  // bool report_cr_est;
  bool verbose;

  // tracking error status
  bool there_is_memerr;
};

typedef struct psz_context psz_context;
typedef psz_context pszctx;

void capi_psz_version();
void capi_psz_versioninfo();

/**
 * @brief Return a pszctx instance with default values.
 *
 * @return pszctx*
 */
pszctx* pszctx_default_values();

/**
 * @brief Modify an empty pszctx with default values.
 *
 */
void pszctx_set_default_values(pszctx*);

/**
 * @brief Use a minimal workset as the return object.
 *
 * @param dtype input data type
 * @param predictor Lorenzo, Spline
 * @param quantizer_radius: int, e.g., 512
 * @param codec HUffman
 * @param eb error bound
 * @param mode Rel, Abs
 * @return pszctx*
 */
pszctx* pszctx_minimal_workset(
    psz_dtype const dtype, psz_predtype const predictor, int const quantizer_radius,
    psz_codectype const codec);

void pszctx_set_rawlen(pszctx* ctx, size_t _x, size_t _y, size_t _z);
void pszctx_set_len(pszctx* ctx, psz_len3 len);
#define get_len3 pszctx_get_len3
psz_len3 pszctx_get_len3(pszctx* ctx);
void pszctx_create_from_argv(pszctx* ctx, int const argc, char** const argv);
void pszctx_create_from_string(pszctx* ctx, const char* in_str, bool dbg_print);

#ifdef __cplusplus
}
#endif

#endif /* A93D242E_0C7C_44BD_AE43_B3A26084971A */
