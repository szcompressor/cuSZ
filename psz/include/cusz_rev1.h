#ifndef PSZ_API_H
#define PSZ_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include "cusz/context.h"

#define DEFAULT_RADIUS 512

typedef struct psz_runtime_config {
  psz_predtype predictor;
  psz_histotype hist;
  psz_codectype codec1;
  psz_codectype _future_codec2;
  psz_mode mode;
  double eb;
  uint16_t radius;
} psz_rc;

// clang-format off
psz_resource* psz_create_resource_manager(psz_dtype t, uint32_t x, uint32_t y, uint32_t z, void* stream);
psz_resource* psz_create_resource_manager_from_CLI(int argc, char** argv, void* stream);
psz_resource* psz_create_resource_manager_from_header(psz_header* header, void* stream);
void psz_modify_resource_manager_from_header(psz_resource* manager, psz_header* header);
int psz_release_resource(psz_resource* manager);
int psz_compress_float(psz_resource* manager, psz_rc rc, float* IN_d_data, psz_header* OUT_compressed_metadata, uint8_t** OUT_dptr_compressed, size_t* OUT_compressed_bytes);
int psz_compress_double(psz_resource* manager, psz_rc rc, double* IN_d_data, psz_header* OUT_compressed_metadata, uint8_t** OUT_dptr_compressed, size_t* OUT_compressed_bytes);
int psz_decompress_float(psz_resource* manager, uint8_t* IN_d_compressed, size_t const IN_compressed_len, float* OUT_d_decompressed);
int psz_decompress_double(psz_resource* manager, uint8_t* IN_d_compressed, size_t const IN_compressed_len, double* OUT_d_decompressed);
// clang-format on

#ifdef __cplusplus
}
#endif

#endif /* PSZ_API_H */
