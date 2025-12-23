%
    module pycusz

    // The original names are not used.
    % rename(version) psz_version;
% rename(versioninfo) psz_versioninfo;
% rename(create) psz_create;
% rename(create_default) psz_create_default;
% rename(create_from_context) psz_create_from_context;
% rename(create_from_header) psz_create_from_header;
% rename(release) psz_release;
% ignore psz_compress;
% rename(decompress) psz_decompress;
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

    % include "context.h" % include "cusz/type.h" % include "header.h" %
    include "cusz.h"

    // directly write python code here
    // REF: https://stackoverflow.com/a/4549685
    // The original names are kept.
    % pythoncode %
{
  Ctx = psz_context Header = psz_header Compressor = psz_compressor Len3 = psz_len3 %
}

extern void psz_version();
extern void psz_versioninfo();
extern psz_compressor* psz_create(
    psz_dtype const, psz_len3 const, psz_predtype const, int const, psz_codectype const);
extern psz_compressor* psz_create_default(psz_dtype const, psz_len3 const);
extern psz_compressor* psz_create_from_context(pszctx* const, psz_len3 const);
extern psz_compressor* psz_create_from_header(psz_header* const);
extern pszerror psz_release(psz_compressor*);
extern pszerror psz_decompress(
    psz_compressor*, uint8_t*, size_t const comp_len, void*, psz_len3 const, void* record,
    void* stream);

% inline %
{
  PyObject* compress(
      psz_compressor * comp, void* uncompressed, psz_len3 const uncomp_len, double const eb,
      psz_mode const mode, void* stream)
  {
    uint8_t* compressed;
    size_t comp_bytes;
    psz_header header;

    pszerror error_code = psz_compress(
        comp, uncompressed, uncomp_len, eb, mode, &compressed, &comp_bytes, &header, NULL, stream);
    PyObject* py_compressed = PyBytes_FromStringAndSize((const char*)compressed, comp_bytes);
    PyObject* py_tuple = Py_BuildValue("(iO)", error_code, py_compressed);
    return py_tuple;
  }
  %
}