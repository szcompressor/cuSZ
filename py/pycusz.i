%module pycusz

// The original names are not used.
%rename(version) capi_psz_version;
%rename(versioninfo) capi_psz_versioninfo;
%rename(create) capi_psz_create;
%rename(create_default) capi_psz_create_default;
%rename(create_from_context) capi_psz_create_from_context;
%rename(release) capi_psz_release;
%rename(compress_init) capi_psz_compress_init;
// %rename(compress) capi_psz_compress; // use a RetType-changing wrapper instead
%ignore capi_psz_compress;
%rename(decompress_init) capi_psz_decompress_init;
%rename(decompress) capi_psz_decompress;


// ignore pszctx related (as of now, 2405)
%ignore pszctx_create_from_argv;
%ignore pszctx_create_from_string;
%ignore pszctx_default_values;
%ignore pszctx_minimal_working_set;
%ignore pszctx_set_default_values;
%ignore pszctx_set_len;
%ignore pszctx_set_rawlen;

%ignore psz_backend;
%ignore psz_device;
%ignore psz_space;
%ignore psz_preprocestype;
%ignore psz_basic_data_description;
%ignore psz_statistic_summary;
%ignore psz_capi_array;
%ignore psz_rettype_archive;
%ignore psz_capi_compact;
%ignore psz_runtime_config;
%ignore psz_timing_mode;

%ignore __F0;
%ignore __I0;
%ignore __U0;

%ignore PSZHEADER_FORCED_ALIGN;
%ignore PSZHEADER_HEADER;
%ignore PSZHEADER_ANCHOR;
%ignore PSZHEADER_VLE;
%ignore PSZHEADER_SPFMT;
%ignore PSZHEADER_END;

%{
#include "context.h"
#include "cusz/type.h"
#include "header.h"
#include "cusz.h"
%}

%include "context.h"
%include "cusz/type.h"
%include "header.h"
%include "cusz.h"

// directly write python code here
// REF: https://stackoverflow.com/a/4549685
// The original names are kept.
%pythoncode %{
Ctx = psz_context
Header = psz_header
Compressor = psz_compressor
Len3 = psz_len3
%}

extern void capi_psz_version();
extern void capi_psz_versioninfo();

extern psz_compressor* capi_psz_create(
    psz_dtype const dtype,  // input
    psz_predtype const predictor, int const quantizer_radius,
    psz_codectype const codec,  // config
    double const eb, psz_mode const mode); // runtime

extern psz_compressor* capi_psz_create_default(
    psz_dtype const dtype, double const eb, psz_mode const mode);

extern psz_compressor* capi_psz_create_from_context(pszctx* const ctx);

extern pszerror capi_psz_release(psz_compressor* comp);

extern pszerror capi_psz_compress_init(psz_compressor* comp, psz_len3 const uncomp_len);

// extern pszerror capi_psz_compress(
//     psz_compressor* comp, void* uncompressed, psz_len3 const uncomp_len,
//     uint8_t** compressed, size_t* comp_bytes, psz_header* header, void* record,
//     void* stream);

extern pszerror capi_psz_decompress_init(psz_compressor* comp, psz_header* header);

extern pszerror capi_psz_decompress(
    psz_compressor* comp, uint8_t* compressed, size_t const comp_len,
    void* decompressed, psz_len3 const decomp_len, void* record, void* stream);


%inline %{
  PyObject* compress(
      psz_compressor* comp, void* uncompressed, psz_len3 const uncomp_len,
      void* stream)
  {
    uint8_t* compressed;
    size_t comp_bytes;
    psz_header header;
  
    pszerror error_code = capi_psz_compress(
        comp, uncompressed, uncomp_len, &compressed, &comp_bytes, &header,
        NULL, stream);
    PyObject* py_compressed = PyBytes_FromStringAndSize((const char*)compressed, comp_bytes);
    PyObject* py_tuple = Py_BuildValue("(iO)", error_code, py_compressed);
    return py_tuple;
  }
%}