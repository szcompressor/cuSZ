#ifndef PSZ_CONTEXT_H
#define PSZ_CONTEXT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "cusz/header.h"
#include "cusz/type.h"

struct psz_cli_config;
typedef struct psz_cli_config psz_cli_config;

struct psz_context;
typedef struct psz_context psz_context;
typedef psz_context psz_ctx;
typedef psz_context psz_manager;
typedef psz_context psz_resource;
typedef psz_context psz_args;

void psz_version();
void psz_versioninfo();
void psz_print_document(bool full);

#ifdef __cplusplus
}
#endif

#endif /* PSZ_CONTEXT_H */
