%module pycuhf

// The original names are not used.
%rename(version) capi_phf_version;
%rename(versioninfo) capi_phf_versioninfo;
%rename(create) capi_phf_create;
%rename(release) capi_phf_release;
%rename(buildbook) capi_phf_buildbook;
%rename(encode) capi_phf_encode;
%rename(decode) capi_phf_decode;

%ignore capi_phf_coarse_tune_sublen;
%ignore capi_phf_coarse_tune;


%{
#include "hf.h"
#include "hf_type.h"
%}

%include "hf.h"
%include "hf_type.h"


extern void capi_phf_version();
extern void capi_phf_versioninfo();
extern phf_codec* capi_phf_create(size_t const inlen, phf_dtype const t, int const bklen);
extern phferr capi_phf_release(phf_codec*);
extern phferr capi_phf_buildbook(phf_codec* codec, uint32_t* d_hist, phf_stream_t);
// capi_phf_encode is redirected to encode below
extern phferr capi_phf_decode(
    phf_codec* codec, uint8_t* encoded, void* decoded, phf_stream_t);

%inline %{
  PyObject* encode(
      phf_codec * codec, void* in, size_t const inlen, phf_stream_t s)
  {
    uint8_t* encoded;
    size_t enc_bytes;

    phferr err = capi_phf_encode(codec, in, inlen, &encoded, &enc_bytes, s);
    PyObject* py_encoded =
        PyBytes_FromStringAndSize((const char*)encoded, enc_bytes);
    PyObject* py_tuple = Py_BuildValue("(iO)", err, py_encoded);
    return py_tuple;
  }
%}