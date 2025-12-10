%module pycuhf

// The original names are not used.
%rename(version) phf_version;
%rename(versioninfo) phf_versioninfo;
%rename(create) phf_create;
%rename(release) phf_release;
%rename(buildbook) phf_buildbook;
%rename(encode) phf_encode;
%rename(decode) phf_decode;

%ignore phf_coarse_tune_sublen;
%ignore phf_coarse_tune;


%{
#include "hf.h"
#include "hf_type.h"
%}

%include "hf.h"
%include "hf_type.h"


extern void phf_version();
extern void phf_versioninfo();
extern phf_codec* phf_create(size_t const inlen, phf_dtype const t, int const bklen);
extern phferr phf_release(phf_codec*);
extern phferr phf_buildbook(phf_codec* codec, uint32_t* d_hist, phf_stream_t);
// phf_encode is redirected to encode below
extern phferr phf_decode(
    phf_codec* codec, uint8_t* encoded, void* decoded, phf_stream_t);

%inline %{
  PyObject* encode(
      phf_codec * codec, void* in, size_t const inlen, phf_stream_t s)
  {
    uint8_t* encoded;
    size_t enc_bytes;

    phferr err = phf_encode(codec, in, inlen, &encoded, &enc_bytes, s);
    PyObject* py_encoded =
        PyBytes_FromStringAndSize((const char*)encoded, enc_bytes);
    PyObject* py_tuple = Py_BuildValue("(iO)", err, py_encoded);
    return py_tuple;
  }
%}