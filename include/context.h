
/**
 * @file context.h
 * @author Jiannan Tian
 * @brief Argument parser (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on: 20-04-24
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef A93D242E_0C7C_44BD_AE43_B3A26084971A
#define A93D242E_0C7C_44BD_AE43_B3A26084971A

#ifdef __cplusplus
extern "C" {
#endif

#include "cusz/type.h"
#include "stdint.h"

struct cusz_context {
    bool task_construct{false};
    bool task_reconstruct{false};
    bool task_dryrun{false};
    bool task_experiment{false};

    bool prep_binning{false};
    bool prep_logtransform{false};
    bool prep_prescan{false};

    bool use_demodata{false};
    bool use_release_input{false};
    bool use_anchor{false};
    bool use_autotune_hf{true};
    bool use_gpu_verify{false};

    bool export_book{false};
    bool export_errctrl{false};

    bool skip_tofile{false};
    bool skip_hf{false};

    bool report_time{false};
    bool report_cr{false};
    bool report_cr_est{false};
    bool verbose{false};

    // sizes
    uint32_t x{1}, y{1}, z{1}, w{1};
    size_t   data_len{1};
    int      ndim{-1};

    // filenames
    char demodata_name[40];
    char infile[500];
    char original_file[500];
    char opath[200];

    // pipeline config
    cusz_dtype dtype{F4};
    cusz_mode  mode{Rel};
    double     eb{0.0};
    int        dict_size{1024}, radius{512};
    int        quant_bytewidth{2}, huff_bytewidth{4};

    // spv gather-scatter config, tmp. unused
    float nz_density{0.2};
    float nz_density_factor{5};

    // codec config
    uint32_t codecs_in_use{0b01};
    int      vle_sublen{512}, vle_pardeg{-1};
};

typedef struct cusz_context cusz_context;
typedef cusz_context        CuszCtx;
typedef cusz_context        PszCtx;

void pszctx_print_document(bool full_document);

void pszctx_parse_argv(cusz_context* ctx, int const argc, char** const argv);
void pszctx_parse_length(cusz_context* ctx, const char* lenstr);
void pszctx_parse_control_string(cusz_context* ctx, const char* in_str, bool dbg_print);
void pszctx_validate(cusz_context* ctx);
void pszctx_load_demo_datasize(cusz_context* ctx, void* demodata_name);

void pszctx_set_rawlen(cusz_context* ctx, size_t _x, size_t _y, size_t _z, size_t _w);
void pszctx_set_len(cusz_context* ctx, cusz_len len);
void pszctx_set_report(cusz_context* ctx, const char* in_str);
void pszctx_set_config(cusz_context* ctx, cusz_config* config);
void pszctx_set_radius(cusz_context* ctx, int _);
void pszctx_set_huffbyte(cusz_context* ctx, int _);
void pszctx_set_huffchunk(cusz_context* ctx, int _);
void pszctx_set_densityfactor(cusz_context* ctx, int _);

void pszctx_create_from_argv(cusz_context* ctx, int const argc, char** const argv);
void pszctx_create_from_string(cusz_context* ctx, const char* in_str, bool dbg_print);

#ifdef __cplusplus
}
#endif

#endif /* A93D242E_0C7C_44BD_AE43_B3A26084971A */
