%
    module pycusz

    // The original names are not used.
    % rename(version) psz_version;
% rename(versioninfo) psz_versioninfo;
% rename(create_resource_manager) psz_create_resource_manager;
% rename(create_resource_manager_from_header) psz_create_resource_manager_from_header;
% rename(release_resource) psz_release_resource;
% rename(compress_float) psz_compress_float;
% rename(decompress_float) psz_decompress_float;
% rename(get_len3) pszctx_get_len3;

// ignore pszctx related (as of now, 2405)
% ignore pszctx_create_from_argv;
% ignore pszctx_create_from_string;
% ignore pszctx_default_values;
% ignore pszctx_minimal_workset;
% ignore pszctx_set_default_values;
% ignore pszctx_set_len;
% ignore pszctx_set_rawlen;

% ignore psz_backend;
% ignore psz_device;
% ignore psz_space;
% ignore psz_preprocestype;
% ignore psz_data_summary;
% ignore psz_statistics;
% ignore psz_capi_array;
% ignore psz_rettype_archive;
% ignore psz_capi_compact;
% ignore psz_runtime_config;
% ignore psz_timing_mode;

% ignore __F0;
% ignore __I0;
% ignore __U0;

% ignore PSZHEADER_FORCED_ALIGN;
% ignore PSZHEADER_HEADER;
% ignore PSZHEADER_ANCHOR;
% ignore PSZHEADER_ENCODED;
% ignore PSZHEADER_SPFMT;
% ignore PSZHEADER_END;

%
    {
#include "cusz.h"
#include "cusz/context.h"
#include "cusz/header.h"
#include "cusz/type.h"
        % }

    % include "context.h" % include "cusz/type.h" % include "header.h" % include "cusz.h" %
    include "cusz_rev1.h"

    // directly write python code here
    // REF: https://stackoverflow.com/a/4549685
    // The original names are kept.
    % pythoncode %
{
  Ctx = psz_context Header = psz_header Resource = psz_resource Len3 = psz_len3 %
}

extern void psz_version();
extern void psz_versioninfo();
extern psz_resource* psz_create_resource_manager(psz_dtype, psz_len, psz_pipeline, void*);
extern psz_resource* psz_create_resource_manager_from_header(psz_header*, void*);
extern int psz_release_resource(psz_resource*);
extern int psz_compress_float(psz_resource*, psz_rc2, float*, psz_header*, uint8_t**, size_t*);
extern int psz_decompress_float(psz_resource*, uint8_t*, size_t const, float*);