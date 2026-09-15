#ifndef PSZ_API_H
#define PSZ_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include "cusz/context.h"

// clang-format off
psz_resource* psz_create_resource_manager(psz_dtype dtype, psz_len len, psz_ppl pipeline, void* stream);
psz_resource* psz_create_resource_manager_from_stages(psz_dtype dtype, psz_len len, psz_predictor p1, psz_codec c1, psz_codec optional_c2, void* stream);
psz_resource* psz_create_resource_manager_from_preset(psz_dtype dtype, psz_len len, psz_preset preset, void* stream);
psz_resource* psz_create_resource_manager_from_CLI(int argc, char** argv, void* stream);
psz_resource* psz_create_resource_manager_from_header(psz_header* header, void* stream);
int psz_release_resource(psz_resource* manager);
// why the last creator on this thread returned NULL; psz_error_string names it.
psz_error_status psz_last_error();
// OUT_d_compressed carries its own psz_header, so it can be written out as it stands.
int psz_compress_float(psz_resource* manager, psz_rc2 rc, float* IN_d_data, psz_header* OUT_header, uint8_t** OUT_d_compressed, size_t* OUT_compressed_bytes);
int psz_compress_double(psz_resource* manager, psz_rc2 rc, double* IN_d_data, psz_header* OUT_header, uint8_t** OUT_d_compressed, size_t* OUT_compressed_bytes);
int psz_compress_analyze_float(psz_resource* manager, psz_rc2 rc, float* IN_d_data, u4* exported_h_hist);
int psz_decompress_float(psz_resource* manager, uint8_t* IN_d_compressed, size_t const IN_compressed_len, float* OUT_d_decompressed);
int psz_decompress_double(psz_resource* manager, uint8_t* IN_d_compressed, size_t const IN_compressed_len, double* OUT_d_decompressed);
// clang-format on

#ifdef __cplusplus
}
#endif

#endif /* PSZ_API_H */
