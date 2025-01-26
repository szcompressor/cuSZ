#ifndef PSZ_API_H
#define PSZ_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include "cusz/context.h"

#define DEFAULT_RADIUS 512

// clang-format off
psz_resource* psz_create_resource_manager(psz_dtype dtype, psz_len len, psz_pipeline pipeline, void* stream);
psz_resource* psz_create_resource_manager_from_CLI(int argc, char** argv, void* stream);
psz_resource* psz_create_resource_manager_from_header(psz_header* header, void* stream);
void psz_modify_resource_manager_from_header(psz_resource* manager, psz_header* header);
int psz_release_resource(psz_resource* manager);
int psz_compress_float(psz_resource* manager, psz_rc2 rc, float* IN_d_data, psz_header* OUT_header, uint8_t** OUT_d_compressed, size_t* OUT_compressed_bytes);
int psz_compress_double(psz_resource* manager, psz_rc2 rc, double* IN_d_data, psz_header* OUT_header, uint8_t** OUT_d_compressed, size_t* OUT_compressed_bytes);
int psz_compress_analyize_float(psz_resource* manager, psz_rc2 rc, float* IN_d_data, u4* exported_h_hist);
int psz_decompress_float(psz_resource* manager, uint8_t* IN_d_compressed, size_t const IN_compressed_len, float* OUT_d_decompressed);
int psz_decompress_double(psz_resource* manager, uint8_t* IN_d_compressed, size_t const IN_compressed_len, double* OUT_d_decompressed);
// clang-format on

#define CAPI_psz_create_resource_manager psz_create_resource_manager
#define CAPI_psz_create_resource_manager_from_CLI psz_create_resource_manager_from_CLI
#define CAPI_psz_create_resource_manager_from_header psz_create_resource_manager_from_header
#define CAPI_psz_modify_resource_manager_from_header psz_modify_resource_manager_from_header
#define CAPI_psz_release_resource psz_release_resource
#define CAPI_psz_compress_float psz_compress_float
#define CAPI_psz_compress_double psz_compress_double
#define CAPI_psz_decompress_float psz_decompress_float
#define CAPI_psz_decompress_double psz_decompress_double

#ifdef __cplusplus
}
#endif

#endif /* PSZ_API_H */
